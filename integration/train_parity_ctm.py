#!/usr/bin/env python3
"""
Training script for CTM parity model - matches original repo settings.

Usage:
    cd /home/spicylemon/Desktop/ctm_nano/integration
    python3 train_parity_ctm.py --quick  # Fast training for testing (5k iters)
    python3 train_parity_ctm.py          # Full training (50k iters)
"""
import sys
import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

# Add paths
INTEGRATION_ROOT = Path(__file__).parent
CTM_ROOT = INTEGRATION_ROOT.parent / "continuous-thought-machines"
sys.path.insert(0, str(CTM_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT))

from models.ctm import ContinuousThoughtMachine
from configs.ctm_config import CTMParityConfig, CONFIGS


class ParityDataset(torch.utils.data.Dataset):
    """Parity dataset matching original CTM repo implementation"""
    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random +1/-1 sequence
        vector = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        vector = vector.float()
        # Compute cumulative parity: count -1s mod 2
        negatives = (vector == -1).to(torch.long)
        cumsum = torch.cumsum(negatives, dim=0)
        target = (cumsum % 2 != 0).to(torch.long)
        return vector, target


def parity_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Original CTM parity loss - CRITICAL for learning!
    
    The key insight: uses BOTH minimum-loss tick AND most-certain tick.
    This provides gradient signal even when the model is uncertain.
    
    Predictions: (B, parity_sequence_length, 2, internal_ticks)
    Certainties: (B, 2, internal_ticks)
    Targets: (B, parity_sequence_length)
    """
    B = predictions.size(0)
    T = predictions.size(-1)  # internal ticks
    
    # Compute losses at ALL internal ticks
    # Shape: (B, parity_sequence_length, internal_ticks)
    targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), T, -1)
    losses = nn.CrossEntropyLoss(reduction='none')(
        predictions.flatten(0, 1),  # (B*seq, 2, T)
        targets_expanded.flatten(0, 1).long()  # (B*seq, T)
    ).reshape(predictions[:, :, 0].shape)  # Back to (B, seq, T)
    
    # Average over sequence dimension -> (B, T)
    losses = losses.mean(1)
    
    # Find tick with minimum loss for each sample
    loss_index_1 = losses.argmin(dim=1)
    
    # Find most certain tick for each sample
    loss_index_2 = certainties[:, 1].argmax(-1)
    
    if not use_most_certain:
        loss_index_2 = torch.full_like(loss_index_2, -1)
    
    # Get losses at both ticks
    batch_indexer = torch.arange(B, device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()
    
    # Average of both - THIS IS KEY
    loss = (loss_minimum_ce + loss_selected) / 2
    
    return loss, loss_index_2


def train(config: CTMParityConfig, args):
    """Train a CTM parity model"""
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on {device}")
    
    # Data
    train_data = ParityDataset(sequence_length=config.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=config.parity_sequence_length, length=10000)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=0
    )
    
    # Model
    model = ContinuousThoughtMachine(
        iterations=config.iterations,
        d_model=config.d_model,
        d_input=config.d_input,
        heads=config.heads,
        n_synch_out=config.n_synch_out,
        n_synch_action=config.n_synch_action,
        synapse_depth=config.synapse_depth,
        memory_length=config.memory_length,
        deep_nlms=config.deep_nlms,
        memory_hidden_dims=config.memory_hidden_dims,
        do_layernorm_nlm=config.do_layernorm_nlm,
        backbone_type=config.backbone_type,
        positional_embedding_type=config.positional_embedding_type,
        out_dims=config.out_dims,
        prediction_reshaper=config.prediction_reshaper,
        dropout=config.dropout,
        neuron_select_type=config.neuron_select_type,
        n_random_pairing_self=config.n_random_pairing_self,
    ).to(device)
    
    # Initialize lazy modules
    dummy = torch.zeros(1, config.parity_sequence_length, device=device)
    with torch.no_grad():
        model(dummy)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # Cosine annealing scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.training_iterations - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(args.training_iterations), desc="Training")
    for step in pbar:
        model.train()
        
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        predictions, certainties, _ = model(x)
        # Reshape: (B, out_dims, T) -> (B, seq_len, 2, T)
        predictions = predictions.reshape(x.size(0), config.parity_sequence_length, 2, -1)
        
        # Loss with both min-loss and most-certain ticks
        loss, where_certain = parity_loss(predictions, certainties, y, use_most_certain=True)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Compute accuracy at most-certain tick
        batch_idx = torch.arange(x.size(0), device=device)
        preds_at_certain = predictions.argmax(2)[batch_idx, :, where_certain]
        acc = (preds_at_certain == y).float().mean().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            'tick': f'{where_certain.float().mean():.1f}±{where_certain.float().std():.1f}'
        })
        
        # Evaluation
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            test_correct = 0
            test_samples = 0
            full_correct = 0
            full_samples = 0
            
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(test_loader):
                    if batch_idx >= args.n_test_batches:
                        break
                    x, y = x.to(device), y.to(device)
                    
                    predictions, certainties, _ = model(x)
                    predictions = predictions.reshape(x.size(0), config.parity_sequence_length, 2, -1)
                    
                    where_certain = certainties[:, 1].argmax(dim=-1)
                    bidx = torch.arange(x.size(0), device=device)
                    preds = predictions.argmax(2)[bidx, :, where_certain]
                    
                    test_correct += (preds == y).float().sum().item()
                    test_samples += y.numel()
                    
                    full_correct += (preds == y).all(dim=1).sum().item()
                    full_samples += y.size(0)
            
            test_acc = test_correct / test_samples if test_samples > 0 else 0
            full_acc = full_correct / full_samples if full_samples > 0 else 0
            
            print(f"\nStep {step}: Test Acc={test_acc:.4f}, Full Seq Acc={full_acc:.4f}")
            
            # Save best
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint_path = checkpoint_dir / "parity_ctm_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'step': step,
                    'test_acc': test_acc,
                    'full_seq_acc': full_acc,
                }, checkpoint_path)
                print(f"Saved best model (acc={test_acc:.4f}) to {checkpoint_path}")
    
    # Save final
    checkpoint_path = checkpoint_dir / "parity_ctm_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': args.training_iterations,
    }, checkpoint_path)
    print(f"Saved final model to {checkpoint_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CTM Parity Model")
    parser.add_argument('--config', type=str, default='parity_small', 
                        choices=list(CONFIGS.keys()), help='Config preset')
    parser.add_argument('--quick', action='store_true', help='Quick training for testing')
    parser.add_argument('--training_iterations', type=int, default=50000, help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--eval_every', type=int, default=1000, help='Eval frequency')
    parser.add_argument('--n_test_batches', type=int, default=20, help='Test batches for eval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint dir')
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.training_iterations = 5000
        args.eval_every = 500
        print("Quick mode: 5000 iterations for testing")
    
    config = CONFIGS[args.config]
    print(f"Using config: {args.config}")
    print(f"  d_model={config.d_model}, iterations={config.iterations}, memory_length={config.memory_length}")
    print(f"  neuron_select_type={config.neuron_select_type}")
    
    train(config, args)


if __name__ == "__main__":
    main()
