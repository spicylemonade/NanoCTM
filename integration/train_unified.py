#!/usr/bin/env python3
"""
Unified Nanochat + CTM Architecture

This is TRUE Phase 2: Real nanochat Transformer with CTM refinement head.

Architecture:
    Input tokens → Nanochat (Transformer) → Hidden states → CTMHead → Refined logits
    
The CTMHead takes the LLM's hidden states and refines them through
iterative "thinking" before producing final predictions.
"""

import sys
import os
from pathlib import Path
import argparse
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
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

from nanochat.gpt import GPT, GPTConfig
from adapters.ctm_head import CTMHead, CTMHeadConfig


# ============ Dataset: Arithmetic in Text ============

class ArithmeticTextDataset(Dataset):
    """
    Text-based arithmetic dataset for testing LLM+CTM.
    
    Examples:
        Input:  "What is 23 + 45? Answer:"
        Target: " 68"
        
        Input:  "Calculate 7 * 8. Result:"
        Target: " 56"
    """
    def __init__(
        self,
        num_samples: int = 10000,
        max_num: int = 100,
        operations: List[str] = None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.max_num = max_num
        self.operations = operations or ['+', '-', '*']
        
        random.seed(seed)
        
        # Simple character-level tokenizer for numbers
        # Vocab: 0-9, +, -, *, =, space, and special tokens
        self.char_to_id = {
            '<pad>': 0, '<bos>': 1, '<eos>': 2,
            ' ': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8,
            '5': 9, '6': 10, '7': 11, '8': 12, '9': 13,
            '+': 14, '-': 15, '*': 16, '=': 17,
            'W': 18, 'h': 19, 'a': 20, 't': 21, 'i': 22, 's': 23,
            '?': 24, 'A': 25, 'n': 26, 'w': 27, 'e': 28, 'r': 29,
            ':': 30, 'C': 31, 'l': 32, 'c': 33, 'u': 34,
            'R': 35, 'o': 36, '.': 37,
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.pad_id = 0
        
        # Generate samples
        self.samples = []
        templates = [
            ("What is {} {} {}? Answer:", " {}"),
            ("Calculate {} {} {}. Result:", " {}"),
            ("{} {} {} =", " {}"),
        ]
        
        for _ in range(num_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            op = random.choice(self.operations)
            
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            else:
                result = a + b
                
            template_in, template_out = random.choice(templates)
            input_text = template_in.format(a, op, b)
            target_text = template_out.format(result)
            
            self.samples.append({
                'input': input_text,
                'target': target_text,
                'full': input_text + target_text,
                'a': a,
                'b': b,
                'op': op,
                'result': result,
            })
    
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.char_to_id.get(c, 3) for c in text]  # default to space
        return tokens
    
    def detokenize(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize full sequence
        full_tokens = self.tokenize(sample['full'])
        input_tokens = self.tokenize(sample['input'])
        
        # For training: input is everything, target is shifted by 1
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        # Mask: only compute loss on the answer part
        input_len = len(input_tokens) - 1  # -1 because we shift
        loss_mask = torch.zeros_like(target_ids, dtype=torch.float)
        loss_mask[input_len:] = 1.0
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'loss_mask': loss_mask,
            'input_text': sample['input'],
            'target_text': sample['target'],
            'result': sample['result'],
        }


def collate_fn(batch):
    """Collate with padding."""
    max_len = max(b['input_ids'].size(0) for b in batch)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 for ignore
    loss_mask = torch.zeros(len(batch), max_len)
    
    for i, b in enumerate(batch):
        seq_len = b['input_ids'].size(0)
        input_ids[i, :seq_len] = b['input_ids']
        target_ids[i, :seq_len] = b['target_ids']
        loss_mask[i, :seq_len] = b['loss_mask']
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'loss_mask': loss_mask,
    }


# ============ Unified Model ============

class UnifiedNanochatCTM(nn.Module):
    """
    Unified Nanochat + CTM architecture.
    
    This is a TRUE hybrid:
    - Nanochat handles token embeddings and self-attention
    - CTM refines the hidden states before projection to logits
    """
    
    def __init__(
        self,
        nanochat_config: GPTConfig,
        ctm_config: CTMHeadConfig,
        freeze_nanochat: bool = False,
        ctm_blend_mode: str = 'replace',  # 'replace', 'add', 'gate'
    ):
        super().__init__()
        
        # Build nanochat
        self.nanochat = GPT(nanochat_config)
        self.nanochat.init_weights()
        
        # Configure CTM to match nanochat dimensions
        ctm_config.model_dim = nanochat_config.n_embd
        ctm_config.vocab_size = nanochat_config.vocab_size
        
        # Build CTM head
        self.ctm_head = CTMHead(ctm_config)
        
        self.freeze_nanochat = freeze_nanochat
        self.blend_mode = ctm_blend_mode
        self.config = nanochat_config
        
        if freeze_nanochat:
            for param in self.nanochat.parameters():
                param.requires_grad = False
    
    def get_nanochat_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from nanochat (before lm_head)."""
        B, T = idx.size()
        
        # Get rotary embeddings
        T0 = 0
        cos_sin = self.nanochat.cos[:, T0:T0+T], self.nanochat.sin[:, T0:T0+T]
        
        # Forward through transformer blocks
        x = self.nanochat.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        
        for block in self.nanochat.transformer.h:
            x = block(x, cos_sin, kv_cache=None)
        
        x = F.rms_norm(x, (x.size(-1),))
        return x
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        use_ctm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through unified model.
        
        Args:
            idx: [B, T] input token ids
            targets: [B, T] target token ids (optional)
            loss_mask: [B, T] mask for loss computation
            use_ctm: Whether to use CTM refinement
            
        Returns:
            dict with logits, loss, etc.
        """
        B, T = idx.size()
        
        # Get hidden states from nanochat
        if self.freeze_nanochat:
            with torch.no_grad():
                hidden = self.get_nanochat_hidden_states(idx)
        else:
            hidden = self.get_nanochat_hidden_states(idx)
        
        # Get base logits from nanochat's lm_head
        base_logits = self.nanochat.lm_head(hidden)
        softcap = 15
        base_logits = softcap * torch.tanh(base_logits / softcap)
        
        if use_ctm:
            # Run CTM refinement
            ctm_result = self.ctm_head(hidden, base_logits)
            
            if self.blend_mode == 'replace':
                logits = ctm_result['logits']
            elif self.blend_mode == 'add':
                logits = base_logits + ctm_result['logits']
            elif self.blend_mode == 'gate':
                logits = ctm_result['mixed_logits']
            else:
                logits = ctm_result['logits']
        else:
            logits = base_logits
            ctm_result = None
        
        result = {
            'logits': logits,
            'base_logits': base_logits,
        }
        
        if ctm_result is not None:
            result['ctm_logits'] = ctm_result['logits']
            result['certainties'] = ctm_result.get('certainties')
        
        # Compute loss if targets provided
        if targets is not None:
            logits_flat = logits.float().view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            if loss_mask is not None:
                # Weighted loss
                loss_per_token = F.cross_entropy(
                    logits_flat, targets_flat, 
                    ignore_index=-1, reduction='none'
                )
                loss_per_token = loss_per_token.view(B, T)
                loss = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
            
            result['loss'] = loss
            
            # Also compute base model loss for comparison
            base_logits_flat = base_logits.float().view(-1, base_logits.size(-1))
            if loss_mask is not None:
                base_loss_per_token = F.cross_entropy(
                    base_logits_flat, targets_flat,
                    ignore_index=-1, reduction='none'
                ).view(B, T)
                base_loss = (base_loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            else:
                base_loss = F.cross_entropy(base_logits_flat, targets_flat, ignore_index=-1)
            result['base_loss'] = base_loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 10,
        use_ctm: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        
        ids = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            result = self.forward(ids, use_ctm=use_ctm)
            next_logits = result['logits'][:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        
        return ids


# ============ Training ============

def evaluate_accuracy(model, dataloader, dataset, device, use_ctm=True):
    """Evaluate exact match accuracy on arithmetic."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            
            # For each sample, generate the answer
            for i in range(input_ids.size(0)):
                # Find where the actual input ends (before padding)
                sample = dataset.samples[total]
                prompt = dataset.tokenize(sample['input'])
                prompt_tensor = torch.tensor([prompt], device=device)
                
                # Generate
                target_len = len(dataset.tokenize(sample['target']))
                output = model.generate(prompt_tensor, max_new_tokens=target_len, use_ctm=use_ctm)
                
                # Decode and check
                generated = output[0, len(prompt):].tolist()
                generated_text = dataset.detokenize(generated).strip()
                target_text = sample['target'].strip()
                
                try:
                    generated_num = int(generated_text)
                    if generated_num == sample['result']:
                        correct += 1
                except:
                    pass
                
                total += 1
                
                if total >= 100:  # Limit evaluation
                    break
            if total >= 100:
                break
    
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Train Unified Nanochat+CTM')
    
    # Model args
    parser.add_argument('--n_layer', type=int, default=4, help='Nanochat layers')
    parser.add_argument('--n_head', type=int, default=4, help='Attention heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dim')
    parser.add_argument('--ctm_dim', type=int, default=128, help='CTM internal dim')
    parser.add_argument('--ctm_iterations', type=int, default=15, help='CTM thinking steps')
    parser.add_argument('--freeze_nanochat', action='store_true', help='Freeze nanochat weights')
    parser.add_argument('--blend_mode', type=str, default='replace', choices=['replace', 'add', 'gate'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=500, help='Validation samples')
    parser.add_argument('--max_num', type=int, default=50, help='Max number in arithmetic')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='checkpoints/unified', help='Save dir')
    parser.add_argument('--no_ctm', action='store_true', help='Train without CTM (baseline)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ArithmeticTextDataset(
        num_samples=args.train_samples,
        max_num=args.max_num,
        operations=['+', '-'],  # Start simple
        seed=42,
    )
    val_dataset = ArithmeticTextDataset(
        num_samples=args.val_samples,
        max_num=args.max_num,
        operations=['+', '-'],
        seed=123,
    )
    
    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Sample: '{train_dataset.samples[0]['full']}'")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    
    # Create model configs
    nanochat_config = GPTConfig(
        vocab_size=train_dataset.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
        sequence_len=64,
    )
    
    ctm_config = CTMHeadConfig(
        model_dim=args.n_embd,
        vocab_size=train_dataset.vocab_size,
        ctm_dim=args.ctm_dim,
        ctm_input_dim=args.ctm_dim // 2,
        iterations=args.ctm_iterations,
        memory_length=8,
        n_synch_out=args.ctm_dim // 2,
        n_synch_action=args.ctm_dim // 2,
        synapse_depth=2,
        n_random_pairing_self=8,
        dropout=0.1,
    )
    
    # Create model
    print("\nCreating Unified Nanochat+CTM model...")
    if args.no_ctm:
        # Baseline: just nanochat
        model = GPT(nanochat_config)
        model.init_weights()
        use_ctm = False
        print("Mode: Nanochat only (baseline)")
    else:
        model = UnifiedNanochatCTM(
            nanochat_config=nanochat_config,
            ctm_config=ctm_config,
            freeze_nanochat=args.freeze_nanochat,
            ctm_blend_mode=args.blend_mode,
        )
        use_ctm = True
        print(f"Mode: Unified (blend={args.blend_mode}, freeze_nanochat={args.freeze_nanochat})")
    
    model = model.to(device)
    
    # Initialize lazy modules
    with torch.no_grad():
        dummy = torch.zeros(1, 10, dtype=torch.long, device=device)
        if args.no_ctm:
            _ = model(dummy)
        else:
            _ = model(dummy, use_ctm=True)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = []
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Base Loss':>10} | {'CTM Gain':>10}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_base_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            optimizer.zero_grad()
            
            if args.no_ctm:
                loss = model(input_ids, targets=target_ids)
                base_loss = loss
            else:
                result = model(input_ids, targets=target_ids, loss_mask=loss_mask, use_ctm=use_ctm)
                loss = result['loss']
                base_loss = result.get('base_loss', loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_base_loss += base_loss.item() if isinstance(base_loss, torch.Tensor) else base_loss
            num_batches += 1
        
        train_loss /= num_batches
        train_base_loss /= num_batches
        
        # Validate
        model.eval()
        val_loss = 0
        val_base_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                loss_mask = batch['loss_mask'].to(device)
                
                if args.no_ctm:
                    loss = model(input_ids, targets=target_ids)
                    base_loss = loss
                else:
                    result = model(input_ids, targets=target_ids, loss_mask=loss_mask, use_ctm=use_ctm)
                    loss = result['loss']
                    base_loss = result.get('base_loss', loss)
                
                val_loss += loss.item()
                val_base_loss += base_loss.item() if isinstance(base_loss, torch.Tensor) else base_loss
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_base_loss /= num_val_batches
        
        scheduler.step()
        
        ctm_gain = val_base_loss - val_loss if not args.no_ctm else 0
        
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_base_loss:>10.4f} | {ctm_gain:>+10.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_base_loss': val_base_loss,
        })
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pt'))
    
    # Final evaluation with generation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    if not args.no_ctm:
        # Test some examples
        print("\nSample generations:")
        for i in range(5):
            sample = val_dataset.samples[i]
            prompt = val_dataset.tokenize(sample['input'])
            prompt_tensor = torch.tensor([prompt], device=device)
            
            # With CTM
            output_ctm = model.generate(prompt_tensor, max_new_tokens=6, use_ctm=True)
            gen_ctm = val_dataset.detokenize(output_ctm[0, len(prompt):].tolist())
            
            # Without CTM
            output_base = model.generate(prompt_tensor, max_new_tokens=6, use_ctm=False)
            gen_base = val_dataset.detokenize(output_base[0, len(prompt):].tolist())
            
            print(f"  Input: {sample['input']}")
            print(f"  Target: {sample['target']}")
            print(f"  Base:   {gen_base}")
            print(f"  CTM:    {gen_ctm}")
            print()
    
    # Save history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to {args.save_dir}/")


if __name__ == "__main__":
    main()

