"""
CTM Tool Wrapper - Exposes trained CTM models as callable tools for nanochat

This is Phase 1: CTM as an external tool that nanochat can invoke.
"""
import sys
import os
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add CTM repo to path
CTM_REPO = Path(__file__).parent.parent.parent / "continuous-thought-machines"
sys.path.insert(0, str(CTM_REPO))

from models.ctm import ContinuousThoughtMachine
from models.utils import compute_normalized_entropy

# Add integration configs
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.ctm_config import CTMParityConfig, IntegrationConfig


class CTMParityTool:
    """
    CTM Parity Tool - computes parity of a bit sequence using CTM's iterative reasoning.
    
    The parity task: given a sequence of bits, output the cumulative parity at each position.
    This is a test case for CTM's ability to do iterative computation.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[CTMParityConfig] = None,
        device: str = "auto"
    ):
        self.config = config or CTMParityConfig()
        
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"CTM Parity Tool using device: {self.device}")
        
        # Build model
        self.model = self._build_model()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint loaded, model has random weights")
            
        self.model.eval()
        
    def _build_model(self) -> ContinuousThoughtMachine:
        """Build the CTM model for parity task"""
        cfg = self.config
        
        model = ContinuousThoughtMachine(
            iterations=cfg.iterations,
            d_model=cfg.d_model,
            d_input=cfg.d_input,
            heads=cfg.heads,
            n_synch_out=cfg.n_synch_out,
            n_synch_action=cfg.n_synch_action,
            synapse_depth=cfg.synapse_depth,
            memory_length=cfg.memory_length,
            deep_nlms=cfg.deep_nlms,
            memory_hidden_dims=cfg.memory_hidden_dims,
            do_layernorm_nlm=cfg.do_layernorm_nlm,
            backbone_type=cfg.backbone_type,
            positional_embedding_type=cfg.positional_embedding_type,
            out_dims=cfg.out_dims,
            prediction_reshaper=cfg.prediction_reshaper,
            dropout=cfg.dropout,
            neuron_select_type=cfg.neuron_select_type,
            n_random_pairing_self=cfg.n_random_pairing_self,
        ).to(self.device)
        
        # Initialize lazy layers with dummy forward pass
        seq_len = cfg.parity_sequence_length
        grid_size = int(math.sqrt(seq_len))
        dummy_input = torch.zeros(1, seq_len, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            model(dummy_input)
            
        return model
    
    def _load_checkpoint(self, path: str):
        """Load model checkpoint"""
        print(f"Loading CTM checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Checkpoint loaded successfully")
        
    def encode_bits(self, bits: List[int]) -> torch.Tensor:
        """
        Encode bit sequence for CTM input.
        Bits should be -1 or 1 (or 0 and 1, will be converted).
        """
        bits = list(bits)
        seq_len = self.config.parity_sequence_length
        
        # Pad or truncate to expected length
        if len(bits) < seq_len:
            bits = bits + [1] * (seq_len - len(bits))  # Pad with 1s (identity for parity)
        elif len(bits) > seq_len:
            bits = bits[:seq_len]
            
        # Convert 0s to -1s if present
        bits = [b if b in [-1, 1] else (1 if b == 1 else -1) for b in bits]
        
        return torch.tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def decode_output(self, predictions: torch.Tensor, certainties: torch.Tensor) -> Tuple[List[int], float, int]:
        """
        Decode CTM output to parity values.
        
        Returns:
            parities: List of cumulative parities at each position
            confidence: Overall confidence score
            tick_used: Which internal tick was used (most certain)
        """
        # predictions shape: [B, seq_len, 2, iterations]
        # certainties shape: [B, 2, iterations]
        
        # Find most certain tick
        # certainties[:, 1] is the "certainty" (1 - normalized_entropy)
        most_certain_tick = certainties[0, 1].argmax().item()
        
        # Get predictions at most certain tick
        pred_at_tick = predictions[0, :, :, most_certain_tick]  # [seq_len, 2]
        parities = pred_at_tick.argmax(dim=1).tolist()  # 0 or 1 for each position
        
        # Convert to -1/1 convention:
        # Dataset: target 0 = positive parity (prod=1), target 1 = negative parity (prod=-1)
        parities = [1 if p == 0 else -1 for p in parities]
        
        confidence = certainties[0, 1, most_certain_tick].item()
        
        return parities, confidence, most_certain_tick
    
    @torch.no_grad()
    def __call__(
        self, 
        bits: List[int], 
        return_all_ticks: bool = False,
        adaptive: bool = True,
        certainty_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute parity of bit sequence using CTM.
        
        Args:
            bits: Input bit sequence (-1s and 1s, or 0s and 1s)
            return_all_ticks: Return predictions at all internal ticks
            adaptive: Stop early if certainty exceeds threshold
            certainty_threshold: Threshold for early stopping
            
        Returns:
            Dictionary with parities, confidence, tick_used, and optionally all_ticks
        """
        x = self.encode_bits(bits)
        
        # Run CTM
        predictions, certainties, sync_out = self.model(x)
        
        # predictions: [B, out_dims, iterations] -> reshape to [B, seq_len, 2, iterations]
        seq_len = self.config.parity_sequence_length
        predictions = predictions.view(1, seq_len, 2, -1)
        
        # Decode
        parities, confidence, tick_used = self.decode_output(predictions, certainties)
        
        result = {
            'parities': parities,
            'confidence': confidence,
            'tick_used': tick_used,
            'total_ticks': self.config.iterations,
        }
        
        if return_all_ticks:
            # Return predictions at all ticks for analysis
            all_parities = []
            all_confidences = []
            for t in range(predictions.shape[-1]):
                pred_at_t = predictions[0, :, :, t].argmax(dim=1).tolist()
                pred_at_t = [1 if p == 0 else -1 for p in pred_at_t]  # 0=positive, 1=negative
                all_parities.append(pred_at_t)
                all_confidences.append(certainties[0, 1, t].item())
            result['all_parities'] = all_parities
            result['all_confidences'] = all_confidences
            
        return result
    
    def compute_ground_truth(self, bits: List[int]) -> List[int]:
        """Compute ground truth cumulative parities"""
        bits = [1 if b in [1] else -1 for b in bits]  # Normalize
        parities = []
        running = 1
        for b in bits:
            running *= b
            parities.append(running)
        return parities


class CTMToolRouter:
    """
    Routes tool calls from nanochat to appropriate CTM models.
    
    Parses tool call syntax like:
        [CTM_PARITY] 1 -1 1 1 -1 ...
        [CTM_MAZE] <maze_data>
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.tools: Dict[str, Any] = {}
        
    def register_tool(self, name: str, tool: Any):
        """Register a CTM tool"""
        self.tools[name.upper()] = tool
        print(f"Registered CTM tool: {name}")
        
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse tool call from text.
        
        Returns (tool_name, arguments) if found, else None.
        """
        for tool_name in self.config.tools:
            marker = f"[CTM_{tool_name}]"
            if marker in text:
                # Extract everything after the marker
                idx = text.find(marker)
                args_text = text[idx + len(marker):].strip()
                # Take until newline or end
                args_text = args_text.split('\n')[0].strip()
                return tool_name, args_text
        return None
        
    def execute(self, tool_name: str, args_text: str) -> str:
        """Execute a tool call and return result string"""
        tool_name = tool_name.upper()
        
        if tool_name not in self.tools:
            return f"[CTM_ERROR] Unknown tool: {tool_name}"
            
        tool = self.tools[tool_name]
        
        try:
            if tool_name == "PARITY":
                # Parse bit sequence
                bits = [int(x) for x in args_text.split()]
                result = tool(bits)
                # Format result
                parity_str = ' '.join(str(p) for p in result['parities'])
                return f"[CTM_RESULT] parities={parity_str} confidence={result['confidence']:.3f} tick={result['tick_used']}"
            else:
                return f"[CTM_ERROR] Tool {tool_name} not implemented"
        except Exception as e:
            return f"[CTM_ERROR] {str(e)}"


# Convenience function for quick testing
def create_parity_tool(checkpoint_path: Optional[str] = None, config_name: str = 'parity_small') -> CTMParityTool:
    """Create a parity tool with specified config"""
    from configs.ctm_config import CONFIGS
    config = CONFIGS.get(config_name, CTMParityConfig())
    return CTMParityTool(checkpoint_path=checkpoint_path, config=config)


if __name__ == "__main__":
    # Quick test
    print("Testing CTM Parity Tool...")
    
    tool = CTMParityTool(config=CTMParityConfig(iterations=25, d_model=256))
    
    # Test with a simple sequence
    test_bits = [1, -1, 1, 1, -1, 1, -1, -1]  # 8 bits
    ground_truth = tool.compute_ground_truth(test_bits)
    
    print(f"Input bits: {test_bits}")
    print(f"Ground truth parities: {ground_truth}")
    
    result = tool(test_bits, return_all_ticks=True)
    print(f"CTM predicted parities: {result['parities'][:len(test_bits)]}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Tick used: {result['tick_used']}/{result['total_ticks']}")
    
    # Check accuracy
    correct = sum(1 for p, gt in zip(result['parities'][:len(test_bits)], ground_truth) if p == gt)
    print(f"Accuracy: {correct}/{len(test_bits)}")

