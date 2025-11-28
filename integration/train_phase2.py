#!/usr/bin/env python3
"""
Phase 2 Training: CTMHead as a Refinement Layer on Nanochat

This script trains a CTMHead module that takes nanochat's hidden states
and refines predictions for reasoning-heavy tasks.

The idea:
- Nanochat handles language understanding (frozen)
- CTMHead does iterative "thinking" to refine outputs
- Together they should outperform either alone on reasoning tasks

Task: Binary addition (multi-step reasoning)
Example: "1011 + 0110 = " → "10001"

This is hard for LLMs because:
- Requires carrying (multi-step reasoning)
- Position-dependent rules
- CTM's iterative computation should help
"""

import sys
import os
from pathlib import Path
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
CTM_ROOT = PROJECT_ROOT / "continuous-thought-machines"
NANOCHAT_ROOT = PROJECT_ROOT / "nanochat"
sys.path.insert(0, str(CTM_ROOT))
sys.path.insert(0, str(NANOCHAT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from adapters.ctm_head import CTMHead, CTMHeadConfig


# ============ Task: Binary Addition ============

class BinaryAdditionDataset(Dataset):
    """
    Dataset for binary addition task.
    
    Examples:
        "1011 + 0110 = 10001"
        "111 + 001 = 1000"
    
    This requires multi-step reasoning with carry propagation.
    """
    def __init__(
        self,
        num_samples: int = 10000,
        num_bits: int = 8,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_bits = num_bits
        
        random.seed(seed)
        
        # Generate samples
        self.samples = []
        for _ in range(num_samples):
            a = random.randint(0, 2**num_bits - 1)
            b = random.randint(0, 2**num_bits - 1)
            result = a + b
            
            # Convert to binary strings
            a_bin = format(a, f'0{num_bits}b')
            b_bin = format(b, f'0{num_bits}b')
            r_bin = format(result, f'0{num_bits+1}b')  # +1 for carry
            
            self.samples.append({
                'a': a_bin,
                'b': b_bin,
                'result': r_bin,
                'a_int': a,
                'b_int': b,
                'result_int': result,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create input embedding (one-hot style)
        # Format: [a_bits, op, b_bits, eq]
        # a_bits: 0/1 for each bit
        # op: special token for +
        # b_bits: 0/1 for each bit  
        # eq: special token for =
        
        a_emb = torch.tensor([float(b) for b in sample['a']])
        b_emb = torch.tensor([float(b) for b in sample['b']])
        r_emb = torch.tensor([float(b) for b in sample['result']])
        
        return {
            'a': a_emb,
            'b': b_emb,
            'result': r_emb,
            'a_int': sample['a_int'],
            'b_int': sample['b_int'],
            'result_int': sample['result_int'],
        }


# ============ Simplified Model for Phase 2 Testing ============

class SimpleHiddenEncoder(nn.Module):
    """
    Simulates nanochat's hidden state output.
    
    In real Phase 2, this would be the frozen nanochat model.
    For testing, we use a simple MLP that learns to encode the binary addition task.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)


class Phase2Model(nn.Module):
    """
    Complete Phase 2 model:
    - Base encoder (simulating nanochat hidden states)
    - CTMHead for refinement
    """
    def __init__(
        self,
        num_bits: int = 8,
        hidden_dim: int = 256,
        ctm_config: Optional[CTMHeadConfig] = None,
        use_ctm: bool = True,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.hidden_dim = hidden_dim
        self.use_ctm = use_ctm
        
        # Input: two binary numbers
        input_dim = num_bits * 2  # a + b concatenated
        output_dim = num_bits + 1  # result with potential carry
        
        # Base encoder (simulating frozen nanochat)
        self.base_encoder = SimpleHiddenEncoder(input_dim, hidden_dim, hidden_dim)
        
        # Base output head (simulating nanochat's lm_head)
        self.base_head = nn.Linear(hidden_dim, output_dim)
        
        if use_ctm and ctm_config is not None:
            # Override config for this task
            ctm_config.model_dim = hidden_dim
            ctm_config.vocab_size = output_dim
            self.ctm_head = CTMHead(ctm_config)
        else:
            self.ctm_head = None
            
        self.output_dim = output_dim
        
    def forward(self, a: torch.Tensor, b: torch.Tensor, use_ctm_override: Optional[bool] = None):
        """
        Forward pass.
        
        Args:
            a: [B, num_bits] binary tensor
            b: [B, num_bits] binary tensor
            use_ctm_override: Override self.use_ctm for this forward
            
        Returns:
            dict with predictions and optionally CTM info
        """
        # Concatenate inputs
        x = torch.cat([a, b], dim=-1)  # [B, num_bits*2]
        
        # Get hidden states from base encoder
        hidden = self.base_encoder(x)  # [B, hidden_dim]
        
        # Base prediction
        base_logits = self.base_head(hidden)  # [B, output_dim]
        
        use_ctm = use_ctm_override if use_ctm_override is not None else self.use_ctm
        
        if use_ctm and self.ctm_head is not None:
            # Reshape for CTMHead: [B, 1, hidden_dim] (sequence of 1)
            hidden_seq = hidden.unsqueeze(1)
            base_logits_seq = base_logits.unsqueeze(1)
            
            ctm_result = self.ctm_head(hidden_seq, base_logits_seq)
            
            # Extract from sequence dimension
            logits = ctm_result['logits'].squeeze(1)
            mixed_logits = ctm_result['mixed_logits'].squeeze(1)
            
            return {
                'logits': logits,
                'base_logits': base_logits,
                'mixed_logits': mixed_logits,
                'certainties': ctm_result['certainties'].squeeze(1),
            }
        else:
            return {
                'logits': base_logits,
                'base_logits': base_logits,
            }


# ============ Training ============

def compute_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute bit-wise accuracy."""
    pred = (pred_logits > 0).float()
    correct = (pred == target).float()
    return correct.mean().item()


def compute_exact_match(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute exact match accuracy (all bits correct)."""
    pred = (pred_logits > 0).float()
    match = (pred == target).all(dim=-1).float()
    return match.mean().item()


def train_epoch(
    model: Phase2Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freeze_base: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    if freeze_base:
        # Freeze base encoder (simulating frozen nanochat)
        for param in model.base_encoder.parameters():
            param.requires_grad = False
        for param in model.base_head.parameters():
            param.requires_grad = False
    
    total_loss = 0
    total_acc = 0
    total_exact = 0
    total_base_acc = 0
    num_batches = 0
    
    for batch in dataloader:
        a = batch['a'].to(device)
        b = batch['b'].to(device)
        target = batch['result'].to(device)
        
        optimizer.zero_grad()
        
        result = model(a, b)
        
        # Binary cross entropy loss on each bit
        loss = F.binary_cross_entropy_with_logits(result['logits'], target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(result['logits'], target)
        total_exact += compute_exact_match(result['logits'], target)
        
        if 'base_logits' in result:
            total_base_acc += compute_accuracy(result['base_logits'], target)
        
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'exact_match': total_exact / num_batches,
        'base_accuracy': total_base_acc / num_batches if total_base_acc > 0 else 0,
    }


@torch.no_grad()
def evaluate(
    model: Phase2Model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    
    total_acc = 0
    total_exact = 0
    total_base_acc = 0
    total_base_exact = 0
    num_batches = 0
    
    for batch in dataloader:
        a = batch['a'].to(device)
        b = batch['b'].to(device)
        target = batch['result'].to(device)
        
        # With CTM
        result = model(a, b, use_ctm_override=True)
        total_acc += compute_accuracy(result['logits'], target)
        total_exact += compute_exact_match(result['logits'], target)
        
        # Without CTM (base only)
        base_result = model(a, b, use_ctm_override=False)
        total_base_acc += compute_accuracy(base_result['logits'], target)
        total_base_exact += compute_exact_match(base_result['logits'], target)
        
        num_batches += 1
    
    return {
        'ctm_accuracy': total_acc / num_batches,
        'ctm_exact_match': total_exact / num_batches,
        'base_accuracy': total_base_acc / num_batches,
        'base_exact_match': total_base_exact / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Train CTMHead refinement')
    parser.add_argument('--num_bits', type=int, default=8, help='Number of bits for binary addition')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--ctm_dim', type=int, default=128, help='CTM internal dimension')
    parser.add_argument('--iterations', type=int, default=15, help='CTM iterations (thinking steps)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=1000, help='Validation samples')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base encoder (true Phase 2)')
    parser.add_argument('--no_ctm', action='store_true', help='Train without CTM (baseline)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/phase2', help='Save directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nCreating datasets (num_bits={args.num_bits})...")
    train_dataset = BinaryAdditionDataset(args.train_samples, args.num_bits, seed=42)
    val_dataset = BinaryAdditionDataset(args.val_samples, args.num_bits, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create CTM config
    ctm_config = CTMHeadConfig(
        model_dim=args.hidden_dim,
        vocab_size=args.num_bits + 1,
        ctm_dim=args.ctm_dim,
        ctm_input_dim=args.ctm_dim // 2,
        iterations=args.iterations,
        memory_length=8,
        n_synch_out=args.ctm_dim // 2,
        n_synch_action=args.ctm_dim // 2,
        synapse_depth=2,
        n_random_pairing_self=8,
        dropout=0.1,
    )
    
    # Create model
    print(f"Creating model (use_ctm={not args.no_ctm})...")
    model = Phase2Model(
        num_bits=args.num_bits,
        hidden_dim=args.hidden_dim,
        ctm_config=ctm_config if not args.no_ctm else None,
        use_ctm=not args.no_ctm,
    )
    model = model.to(device)
    
    # Initialize lazy modules with dummy forward
    with torch.no_grad():
        dummy_a = torch.zeros(1, args.num_bits).to(device)
        dummy_b = torch.zeros(1, args.num_bits).to(device)
        _ = model(dummy_a, dummy_b)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_exact_match = 0
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Acc':>6} | {'Exact':>6} | {'Base':>6} | {'CTM':>6} | {'Base EM':>7} | {'CTM EM':>7}")
    print("-" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.freeze_base)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # Log
        print(f"{epoch:>5} | {train_metrics['loss']:>8.4f} | "
              f"{train_metrics['accuracy']*100:>5.1f}% | "
              f"{train_metrics['exact_match']*100:>5.1f}% | "
              f"{val_metrics['base_accuracy']*100:>5.1f}% | "
              f"{val_metrics['ctm_accuracy']*100:>5.1f}% | "
              f"{val_metrics['base_exact_match']*100:>6.1f}% | "
              f"{val_metrics['ctm_exact_match']*100:>6.1f}%")
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })
        
        # Save best
        if val_metrics['ctm_exact_match'] > best_exact_match:
            best_exact_match = val_metrics['ctm_exact_match']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': {
                    'num_bits': args.num_bits,
                    'hidden_dim': args.hidden_dim,
                    'ctm_dim': args.ctm_dim,
                    'iterations': args.iterations,
                },
            }, os.path.join(args.save_dir, 'best_model.pt'))
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    final_metrics = evaluate(model, val_loader, device)
    print(f"Base Model Only:")
    print(f"  - Bit Accuracy: {final_metrics['base_accuracy']*100:.1f}%")
    print(f"  - Exact Match:  {final_metrics['base_exact_match']*100:.1f}%")
    print(f"\nWith CTM Refinement:")
    print(f"  - Bit Accuracy: {final_metrics['ctm_accuracy']*100:.1f}%")
    print(f"  - Exact Match:  {final_metrics['ctm_exact_match']*100:.1f}%")
    print(f"\nImprovement from CTM:")
    print(f"  - Bit Accuracy: +{(final_metrics['ctm_accuracy']-final_metrics['base_accuracy'])*100:.1f}%")
    print(f"  - Exact Match:  +{(final_metrics['ctm_exact_match']-final_metrics['base_exact_match'])*100:.1f}%")
    
    # Save history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to {args.save_dir}/")


if __name__ == "__main__":
    main()

