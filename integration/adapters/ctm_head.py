"""
CTMHead Adapter - Plugs CTM into nanochat's hidden state space.

This is Phase 2 from the integration plan:
- Freeze nanochat weights
- Extract hidden states from nanochat
- Pass through a CTM refinement module  
- Output refined logits for reasoning-heavy tasks

The idea: let nanochat handle language understanding, let CTM handle
iterative "thinking" over the semantic representation.
"""
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add CTM repo to path
CTM_ROOT = Path(__file__).parent.parent.parent / "continuous-thought-machines"
NANOCHAT_ROOT = Path(__file__).parent.parent.parent / "nanochat"
sys.path.insert(0, str(CTM_ROOT))
sys.path.insert(0, str(NANOCHAT_ROOT))

from models.ctm import ContinuousThoughtMachine
from models.modules import SynapseUNET, SuperLinear, Squeeze


@dataclass
class CTMHeadConfig:
    """Config for CTM refinement head"""
    # Input from nanochat
    model_dim: int = 768  # nanochat's n_embd
    
    # CTM internal dimensions
    ctm_dim: int = 512  # CTM's d_model
    ctm_input_dim: int = 256  # CTM's d_input (for attention)
    
    # CTM architecture
    iterations: int = 25  # Internal thought ticks
    memory_length: int = 8  # Pre-activation history
    heads: int = 4  # Attention heads
    n_synch_out: int = 64  # Output synchronization neurons
    n_synch_action: int = 64  # Action synchronization neurons
    synapse_depth: int = 1  # Synapse model depth
    deep_nlms: bool = True  # Use deep neuron-level models
    memory_hidden_dims: int = 16
    n_random_pairing_self: int = 8
    
    # Output
    vocab_size: int = 50304  # nanochat's vocab size
    
    # Training
    dropout: float = 0.1


class CTMHead(nn.Module):
    """
    CTM Refinement Head for nanochat.
    
    Takes hidden states from nanochat and refines them through CTM's
    iterative reasoning process before projecting to logits.
    
    This can be used to:
    1. Replace nanochat's lm_head for reasoning-heavy domains
    2. Add CTM reasoning on top of nanochat's predictions  
    3. Gate between nanochat and CTM based on uncertainty
    """
    
    def __init__(self, config: CTMHeadConfig):
        super().__init__()
        self.config = config
        
        # Input projection: nanochat hidden dim -> CTM input space
        # We treat the hidden state as a "feature" that CTM attends to
        self.in_proj = nn.Sequential(
            nn.Linear(config.model_dim, config.ctm_input_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ctm_input_dim * 2, config.ctm_input_dim),
            nn.LayerNorm(config.ctm_input_dim),
        )
        
        # Build a simplified CTM core without backbone
        # (we're getting features from nanochat instead)
        self.ctm = self._build_ctm_core()
        
        # Output projection: CTM synchronization -> vocab logits
        self.out_proj = nn.Sequential(
            nn.Linear(config.n_synch_out, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.vocab_size),
        )
        
        # Optional: learnable mixing weight with base model
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
    def _build_ctm_core(self) -> ContinuousThoughtMachine:
        """Build CTM without a backbone (we'll inject features directly)"""
        cfg = self.config
        
        # Use 'none' backbone since we're providing features externally
        ctm = ContinuousThoughtMachine(
            iterations=cfg.iterations,
            d_model=cfg.ctm_dim,
            d_input=cfg.ctm_input_dim,
            heads=cfg.heads,
            n_synch_out=cfg.n_synch_out,
            n_synch_action=cfg.n_synch_action,
            synapse_depth=cfg.synapse_depth,
            memory_length=cfg.memory_length,
            deep_nlms=cfg.deep_nlms,
            memory_hidden_dims=cfg.memory_hidden_dims,
            do_layernorm_nlm=False,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=cfg.n_synch_out,  # Output dim = synch size for now
            prediction_reshaper=[cfg.n_synch_out],
            dropout=cfg.dropout,
            neuron_select_type='random-pairing',
            n_random_pairing_self=cfg.n_random_pairing_self,
        )
        
        return ctm
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_logits: Optional[torch.Tensor] = None,
        return_all_ticks: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CTM head.
        
        Args:
            hidden_states: [B, T, model_dim] from nanochat
            base_logits: [B, T, vocab] optional base model logits for mixing
            return_all_ticks: Return predictions at all CTM ticks
            
        Returns:
            Dictionary with:
                - logits: [B, T, vocab] refined logits
                - certainties: [B, 2, iterations] per-tick certainties
                - mixed_logits: [B, T, vocab] if base_logits provided
        """
        B, T, D = hidden_states.shape
        
        # We'll process each position independently through CTM
        # This is a simplification - could also process sequences
        
        # Project to CTM input space
        h_proj = self.in_proj(hidden_states)  # [B, T, ctm_input_dim]
        
        # Reshape for CTM: treat each position as a separate batch
        # CTM expects: [batch, seq_len, features]
        # We'll use seq_len=1 (single "token" per position)
        h_flat = h_proj.view(B * T, 1, -1)  # [B*T, 1, ctm_input_dim]
        
        # Unfortunately, the vanilla CTM expects image-like inputs for backbone
        # We need to inject features differently - let's bypass backbone
        
        # Direct approach: manually run CTM's core loop with our features
        logits, certainties = self._run_ctm_on_features(h_flat)
        
        # Reshape back to sequence
        logits = logits.view(B, T, -1)  # [B, T, vocab]
        certainties = certainties.view(B, T, 2, -1)  # [B, T, 2, iterations]
        
        result = {
            'logits': logits,
            'certainties': certainties[:, :, :, -1],  # Last tick certainties
        }
        
        # Mix with base logits if provided
        if base_logits is not None:
            alpha = torch.sigmoid(self.mix_weight)
            result['mixed_logits'] = (1 - alpha) * base_logits + alpha * logits
            
        return result
    
    def _run_ctm_on_features(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run CTM core on pre-computed features.
        
        This is a simplified version that manually handles the CTM loop
        since we're bypassing the backbone.
        """
        B = features.size(0)
        device = features.device
        cfg = self.config
        
        # Use features as key-value for attention
        kv = self.ctm.kv_proj(features)  # [B, 1, d_input]
        
        # Initialize CTM state
        state_trace = self.ctm.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.ctm.start_activated_state.unsqueeze(0).expand(B, -1)
        
        # Prepare outputs
        predictions = torch.empty(B, cfg.n_synch_out, cfg.iterations, device=device)
        certainties = torch.empty(B, 2, cfg.iterations, device=device)
        
        # Sync decay params
        decay_alpha_action, decay_beta_action = None, None
        self.ctm.decay_params_action.data = torch.clamp(self.ctm.decay_params_action, 0, 15)
        self.ctm.decay_params_out.data = torch.clamp(self.ctm.decay_params_out, 0, 15)
        r_action = torch.exp(-self.ctm.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.ctm.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        _, decay_alpha_out, decay_beta_out = self.ctm.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )
        
        # Main recurrent loop
        for t in range(cfg.iterations):
            # Compute action synchronization
            sync_action, decay_alpha_action, decay_beta_action = self.ctm.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # Attend to features
            q = self.ctm.q_proj(sync_action).unsqueeze(1)
            attn_out, _ = self.ctm.attention(q, kv, kv)
            attn_out = attn_out.squeeze(1)
            
            # Apply synapses
            pre_synapse = torch.cat([attn_out, activated_state], dim=-1)
            state = self.ctm.synapses(pre_synapse)
            state_trace = torch.cat([state_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1)
            
            # Apply NLMs
            activated_state = self.ctm.trace_processor(state_trace)
            
            # Compute output synchronization
            sync_out, decay_alpha_out, decay_beta_out = self.ctm.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Get predictions
            current_pred = self.ctm.output_projector(sync_out)
            current_certainty = self.ctm.compute_certainty(current_pred)
            
            predictions[:, :, t] = current_pred
            certainties[:, :, t] = current_certainty
        
        # Project sync output to logits using our projection
        final_sync = sync_out
        logits = self.out_proj(final_sync)
        
        return logits, certainties


class HybridNanochatCTM(nn.Module):
    """
    Hybrid model that combines nanochat with CTM reasoning.
    
    Usage modes:
    1. reasoning_only: Only use CTM head for output
    2. mixed: Blend CTM and base model predictions
    3. gated: Use CTM when base model is uncertain
    """
    
    def __init__(
        self,
        nanochat_model: nn.Module,
        ctm_head_config: CTMHeadConfig,
        mode: str = 'mixed',
        freeze_base: bool = True,
    ):
        super().__init__()
        self.nanochat = nanochat_model
        self.ctm_head = CTMHead(ctm_head_config)
        self.mode = mode
        
        if freeze_base:
            for param in self.nanochat.parameters():
                param.requires_grad = False
                
        # Uncertainty threshold for gated mode
        self.uncertainty_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        """
        B, T = idx.shape
        
        # Get nanochat hidden states and logits
        # We need to modify this to get hidden states - assuming modified forward
        with torch.no_grad() if not self.nanochat.training else torch.enable_grad():
            # Standard nanochat forward
            base_logits = self.nanochat(idx)  # [B, T, vocab]
        
        # For hidden states, we need to hook into nanochat's forward
        # This is a simplified version - in practice you'd modify nanochat
        # to return hidden states
        hidden_states = self._get_hidden_states(idx)
        
        # Run CTM head
        ctm_result = self.ctm_head(hidden_states, base_logits)
        
        if self.mode == 'reasoning_only':
            logits = ctm_result['logits']
        elif self.mode == 'mixed':
            logits = ctm_result['mixed_logits']
        elif self.mode == 'gated':
            # Use CTM when base model is uncertain
            base_entropy = self._compute_entropy(base_logits)
            use_ctm = base_entropy > self.uncertainty_threshold
            logits = torch.where(
                use_ctm.unsqueeze(-1),
                ctm_result['logits'],
                base_logits
            )
        else:
            logits = base_logits
            
        result = {
            'logits': logits,
            'base_logits': base_logits,
            'ctm_logits': ctm_result['logits'],
            'certainties': ctm_result['certainties'],
        }
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            result['loss'] = loss
            
        return result
    
    def _get_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden states from nanochat.
        
        This is a simplified implementation - in practice you'd modify
        nanochat's forward to return hidden states directly.
        """
        # Get embeddings
        x = self.nanochat.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        
        # Process through transformer blocks
        T = idx.size(1)
        cos_sin = self.nanochat.cos[:, :T], self.nanochat.sin[:, :T]
        
        for block in self.nanochat.transformer.h:
            x = block(x, cos_sin, kv_cache=None)
            
        x = F.rms_norm(x, (x.size(-1),))
        return x
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy of logit distribution"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        # Normalize by max entropy
        max_entropy = math.log(logits.size(-1))
        return entropy / max_entropy


def create_ctm_head_for_nanochat(
    nanochat_config: Any,
    ctm_iterations: int = 25,
    ctm_dim: int = 512,
) -> CTMHead:
    """
    Create a CTM head configured for a specific nanochat model.
    """
    config = CTMHeadConfig(
        model_dim=nanochat_config.n_embd,
        vocab_size=nanochat_config.vocab_size,
        ctm_dim=ctm_dim,
        iterations=ctm_iterations,
    )
    return CTMHead(config)


if __name__ == "__main__":
    # Test the CTM head
    print("Testing CTM Head...")
    
    config = CTMHeadConfig(
        model_dim=768,
        vocab_size=50304,
        ctm_dim=256,
        iterations=10,
        memory_length=4,
        n_synch_out=64,
        n_synch_action=64,
        n_random_pairing_self=8,
    )
    
    head = CTMHead(config)
    
    # Simulate nanochat hidden states
    B, T, D = 2, 16, 768
    hidden = torch.randn(B, T, D)
    base_logits = torch.randn(B, T, 50304)
    
    # Initialize lazy layers
    with torch.no_grad():
        dummy = torch.randn(1, 1, 256)
        head.ctm.kv_proj(dummy)
    
    result = head(hidden, base_logits)
    
    print(f"Input hidden states: {hidden.shape}")
    print(f"Output logits: {result['logits'].shape}")
    print(f"Mixed logits: {result['mixed_logits'].shape}")
    print(f"Certainties: {result['certainties'].shape}")
    
    n_params = sum(p.numel() for p in head.parameters())
    print(f"CTM Head parameters: {n_params:,}")

