#!/usr/bin/env python3
"""
Ablation Study: Effect of CTM Iterations on Performance

This script runs experiments varying the number of CTM iterations
to show the relationship between "thinking time" and accuracy.
"""

import sys
import os
from pathlib import Path
import json
import random
from typing import List, Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAS_MATPLOTLIB = True
    
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 13
    rcParams['figure.figsize'] = (6, 4)
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
except ImportError:
    HAS_MATPLOTLIB = False

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "continuous-thought-machines"))
sys.path.insert(0, str(PROJECT_ROOT / "nanochat"))
sys.path.insert(0, str(PROJECT_ROOT / "integration"))

from nanochat.gpt import GPT, GPTConfig
from adapters.ctm_head import CTMHead, CTMHeadConfig


class MultiParityDataset(Dataset):
    """Multi-digit parity task."""
    def __init__(self, num_samples=5000, seq_len=5, max_num=20, seed=42):
        random.seed(seed)
        self.seq_len = seq_len
        self.samples = []
        self.char_to_id = {'<pad>': 0, ' ': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
                          '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, '>': 12}
        self.vocab_size = len(self.char_to_id)
        
        for _ in range(num_samples):
            nums = [random.randint(0, max_num) for _ in range(seq_len)]
            parities = [n % 2 for n in nums]
            input_str = ' '.join(str(n) for n in nums) + ' >'
            self.samples.append({'input': input_str, 'parities': parities})
    
    def tokenize(self, s): return [self.char_to_id.get(c, 1) for c in s]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids': torch.tensor(self.tokenize(s['input']), dtype=torch.long),
            'target': torch.tensor(s['parities'], dtype=torch.float)
        }


def collate_fn(batch):
    max_len = max(b['input_ids'].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, :b['input_ids'].size(0)] = b['input_ids']
    return {'input_ids': input_ids, 'target': torch.stack([b['target'] for b in batch])}


class UnifiedModel(nn.Module):
    def __init__(self, vocab_size, output_dim, n_embd=128, n_layer=4, n_head=4, 
                 ctm_iterations=20, ctm_dim=128):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                          n_kv_head=n_head, n_embd=n_embd, sequence_len=64)
        self.nanochat = GPT(config)
        self.nanochat.init_weights()
        self.base_head = nn.Linear(n_embd, output_dim)
        
        ctm_config = CTMHeadConfig(
            model_dim=n_embd, vocab_size=output_dim, ctm_dim=ctm_dim,
            ctm_input_dim=ctm_dim//2, iterations=ctm_iterations, memory_length=8,
            n_synch_out=ctm_dim//2, n_synch_action=ctm_dim//2,
            synapse_depth=2, n_random_pairing_self=8, dropout=0.1,
        )
        self.ctm_head = CTMHead(ctm_config)
        self.n_embd = n_embd
    
    def forward(self, idx, use_ctm=True):
        B, T = idx.size()
        cos_sin = self.nanochat.cos[:, :T], self.nanochat.sin[:, :T]
        x = self.nanochat.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.nanochat.transformer.h:
            x = block(x, cos_sin, kv_cache=None)
        x = F.rms_norm(x, (x.size(-1),))
        
        last_hidden = x[:, -1:, :]
        base_logits = self.base_head(last_hidden[:, 0, :])
        
        if use_ctm:
            result = self.ctm_head(last_hidden, base_logits.unsqueeze(1))
            return result['logits'].squeeze(1), base_logits
        return base_logits, base_logits


def train_and_evaluate(ctm_iterations, config, device):
    """Train model with given CTM iterations and return final metrics."""
    
    train_ds = MultiParityDataset(config['train_samples'], config['seq_len'], config['max_num'], 42)
    val_ds = MultiParityDataset(config['val_samples'], config['seq_len'], config['max_num'], 123)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    model = UnifiedModel(
        vocab_size=train_ds.vocab_size,
        output_dim=config['seq_len'],
        n_embd=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        ctm_iterations=ctm_iterations,
        ctm_dim=config['ctm_dim'],
    )
    model = model.to(device)
    
    with torch.no_grad():
        _ = model(torch.zeros(1, 10, dtype=torch.long, device=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target = batch['target'].to(device)
            optimizer.zero_grad()
            ctm_out, _ = model(input_ids, use_ctm=True)
            loss = F.binary_cross_entropy_with_logits(ctm_out, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    base_correct = ctm_correct = total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            target = batch['target'].to(device)
            ctm_out, base_out = model(input_ids, use_ctm=True)
            base_correct += ((base_out > 0).float() == target).all(dim=1).sum().item()
            ctm_correct += ((ctm_out > 0).float() == target).all(dim=1).sum().item()
            total += target.size(0)
    
    return {
        'base_acc': base_correct / total,
        'ctm_acc': ctm_correct / total,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = {
        'train_samples': 5000,
        'val_samples': 500,
        'seq_len': 5,
        'max_num': 30,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'ctm_dim': 128,
        'epochs': 25,
        'batch_size': 64,
        'lr': 0.001,
    }
    
    # Test different iteration counts
    iteration_counts = [1, 5, 10, 15, 20, 25, 30]
    results = []
    
    print("\n" + "="*60)
    print("ABLATION STUDY: CTM Iterations vs Performance")
    print("="*60)
    print(f"\n{'Iterations':>10} | {'Base Acc':>10} | {'CTM Acc':>10} | {'Improvement':>12}")
    print("-" * 50)
    
    for n_iter in iteration_counts:
        print(f"\nTraining with {n_iter} CTM iterations...")
        metrics = train_and_evaluate(n_iter, config, device)
        results.append({
            'iterations': n_iter,
            **metrics
        })
        
        improvement = metrics['ctm_acc'] - metrics['base_acc']
        print(f"{n_iter:>10} | {metrics['base_acc']*100:>9.1f}% | {metrics['ctm_acc']*100:>9.1f}% | {improvement*100:>+11.1f}%")
    
    # Save results
    save_dir = Path(__file__).parent / 'results' / 'ablation'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'iteration_ablation.json', 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)
    
    # Generate figure
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        iters = [r['iterations'] for r in results]
        base_accs = [r['base_acc']*100 for r in results]
        ctm_accs = [r['ctm_acc']*100 for r in results]
        
        ax.plot(iters, base_accs, 'o--', color='#E74C3C', linewidth=2, 
                markersize=8, label='Base Model')
        ax.plot(iters, ctm_accs, 's-', color='#2ECC71', linewidth=2, 
                markersize=8, label='With CTM')
        
        ax.fill_between(iters, base_accs, ctm_accs, alpha=0.2, color='#2ECC71')
        
        ax.set_xlabel('CTM Iterations (Thinking Steps)')
        ax.set_ylabel('Sequence Accuracy (%)')
        ax.set_title('Effect of CTM Iterations on Performance')
        ax.legend(loc='lower right')
        ax.set_xlim(0, max(iters) + 2)
        ax.set_ylim(0, 105)
        
        plt.savefig(save_dir / 'iteration_ablation.pdf')
        plt.savefig(save_dir / 'iteration_ablation.png')
        plt.close()
        
        print(f"\nFigure saved to {save_dir / 'iteration_ablation.pdf'}")
    
    # Generate LaTeX table
    latex = r"""
\begin{table}[h]
\centering
\caption{Effect of CTM Iterations on Sequence Accuracy}
\label{tab:ablation}
\begin{tabular}{rrrr}
\toprule
\textbf{Iterations} & \textbf{Base Model} & \textbf{With CTM} & \textbf{Improvement} \\
\midrule
"""
    for r in results:
        imp = r['ctm_acc'] - r['base_acc']
        latex += f"{r['iterations']} & {r['base_acc']*100:.1f}\\% & {r['ctm_acc']*100:.1f}\\% & +{imp*100:.1f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()

