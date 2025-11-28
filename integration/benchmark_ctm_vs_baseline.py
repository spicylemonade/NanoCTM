#!/usr/bin/env python3
"""
Benchmark script to compare:
1. Nanochat alone on algorithmic tasks
2. Nanochat + CTM tools
3. CTM standalone accuracy

This measures the gains from using CTM as a thinking coprocessor.

Usage:
    cd /home/spicylemon/Desktop/ctm_nano/integration
    python3 benchmark_ctm_vs_baseline.py --task parity --samples 100
"""
import sys
import time
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add paths
INTEGRATION_ROOT = Path(__file__).parent
NANOCHAT_ROOT = INTEGRATION_ROOT.parent / "nanochat"
CTM_ROOT = INTEGRATION_ROOT.parent / "continuous-thought-machines"

sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(NANOCHAT_ROOT))
sys.path.insert(0, str(CTM_ROOT))

import numpy as np
import torch

from configs.ctm_config import CTMParityConfig, CONFIGS
from tools.ctm_tool import CTMParityTool


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    task: str
    method: str
    samples: int
    accuracy: float
    per_position_accuracy: float
    full_sequence_accuracy: float
    avg_time_ms: float
    config: Dict[str, Any]


class ParityBenchmark:
    """Benchmark parity computation"""
    
    def __init__(self, sequence_length: int = 64, seed: int = 42):
        self.sequence_length = sequence_length
        np.random.seed(seed)
        
    def generate_sample(self) -> tuple:
        """Generate a single parity sample"""
        bits = np.random.choice([-1, 1], size=self.sequence_length)
        parities = np.cumprod(bits)
        return bits.tolist(), parities.tolist()
    
    def generate_batch(self, n: int) -> List[tuple]:
        """Generate n samples"""
        return [self.generate_sample() for _ in range(n)]
    
    def compute_accuracy(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]]
    ) -> tuple:
        """Compute per-position and full-sequence accuracy"""
        total_correct = 0
        total_positions = 0
        full_correct = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # Per-position
            for p, g in zip(pred, gt):
                if p == g:
                    total_correct += 1
                total_positions += 1
            # Full sequence
            if pred == gt:
                full_correct += 1
                
        per_pos_acc = total_correct / total_positions if total_positions > 0 else 0
        full_seq_acc = full_correct / len(predictions) if predictions else 0
        
        return per_pos_acc, full_seq_acc


def benchmark_ctm_standalone(
    samples: List[tuple],
    config: CTMParityConfig,
    checkpoint_path: Optional[str] = None
) -> BenchmarkResult:
    """Benchmark CTM on parity task"""
    
    tool = CTMParityTool(config=config, checkpoint_path=checkpoint_path)
    benchmark = ParityBenchmark(config.parity_sequence_length)
    
    predictions = []
    times = []
    
    for bits, gt in samples:
        start = time.time()
        result = tool(bits)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # ms
        
        # Truncate predictions to input length
        pred = result['parities'][:len(bits)]
        predictions.append(pred)
    
    ground_truths = [gt for _, gt in samples]
    per_pos_acc, full_seq_acc = benchmark.compute_accuracy(predictions, ground_truths)
    
    return BenchmarkResult(
        task="parity",
        method="ctm_standalone",
        samples=len(samples),
        accuracy=per_pos_acc,
        per_position_accuracy=per_pos_acc,
        full_sequence_accuracy=full_seq_acc,
        avg_time_ms=np.mean(times),
        config={
            'd_model': config.d_model,
            'iterations': config.iterations,
            'memory_length': config.memory_length,
        }
    )


def benchmark_naive_baseline(
    samples: List[tuple]
) -> BenchmarkResult:
    """Benchmark naive approach (just guessing randomly)"""
    
    benchmark = ParityBenchmark()
    predictions = []
    
    for bits, gt in samples:
        # Random guess
        pred = [np.random.choice([-1, 1]) for _ in bits]
        predictions.append(pred)
    
    ground_truths = [gt for _, gt in samples]
    per_pos_acc, full_seq_acc = benchmark.compute_accuracy(predictions, ground_truths)
    
    return BenchmarkResult(
        task="parity",
        method="random_baseline",
        samples=len(samples),
        accuracy=per_pos_acc,
        per_position_accuracy=per_pos_acc,
        full_sequence_accuracy=full_seq_acc,
        avg_time_ms=0.01,  # Negligible
        config={}
    )


def benchmark_python_baseline(
    samples: List[tuple]
) -> BenchmarkResult:
    """Benchmark direct Python computation (optimal baseline)"""
    
    benchmark = ParityBenchmark()
    predictions = []
    times = []
    
    for bits, gt in samples:
        start = time.time()
        # Direct computation
        pred = []
        running = 1
        for b in bits:
            running *= b
            pred.append(running)
        elapsed = time.time() - start
        times.append(elapsed * 1000)
        predictions.append(pred)
    
    ground_truths = [gt for _, gt in samples]
    per_pos_acc, full_seq_acc = benchmark.compute_accuracy(predictions, ground_truths)
    
    return BenchmarkResult(
        task="parity",
        method="python_direct",
        samples=len(samples),
        accuracy=per_pos_acc,
        per_position_accuracy=per_pos_acc,
        full_sequence_accuracy=full_seq_acc,
        avg_time_ms=np.mean(times),
        config={}
    )


def print_results(results: List[BenchmarkResult]):
    """Pretty print benchmark results"""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    headers = ["Method", "Per-Pos Acc", "Full-Seq Acc", "Time (ms)"]
    row_fmt = "{:<20} {:>12} {:>12} {:>12}"
    
    print(row_fmt.format(*headers))
    print("-" * 70)
    
    for r in results:
        print(row_fmt.format(
            r.method,
            f"{r.per_position_accuracy:.2%}",
            f"{r.full_sequence_accuracy:.2%}",
            f"{r.avg_time_ms:.2f}"
        ))
    
    print("=" * 70)
    
    # Analysis
    print("\nAnalysis:")
    ctm_result = next((r for r in results if 'ctm' in r.method), None)
    random_result = next((r for r in results if 'random' in r.method), None)
    python_result = next((r for r in results if 'python' in r.method), None)
    
    if ctm_result and random_result:
        improvement = ctm_result.per_position_accuracy - random_result.per_position_accuracy
        print(f"  CTM vs Random: {improvement:+.2%} per-position accuracy")
    
    if ctm_result and python_result:
        gap = python_result.per_position_accuracy - ctm_result.per_position_accuracy
        print(f"  CTM vs Optimal: {gap:-.2%} gap to optimal")
        slowdown = ctm_result.avg_time_ms / max(python_result.avg_time_ms, 0.001)
        print(f"  CTM slowdown: {slowdown:.1f}x vs Python")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CTM vs baselines")
    parser.add_argument('--task', type=str, default='parity', choices=['parity'])
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--config', type=str, default='parity_small', choices=list(CONFIGS.keys()))
    parser.add_argument('--checkpoint', type=str, default=None, help='CTM checkpoint path')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()
    
    print(f"Benchmarking {args.task} task with {args.samples} samples")
    print(f"Config: {args.config}, Sequence length: {args.seq_length}")
    
    # Generate samples
    benchmark = ParityBenchmark(args.seq_length)
    samples = benchmark.generate_batch(args.samples)
    print(f"Generated {len(samples)} samples")
    
    results = []
    
    # Random baseline
    print("\nRunning random baseline...")
    results.append(benchmark_naive_baseline(samples))
    
    # Python optimal
    print("Running Python direct computation...")
    results.append(benchmark_python_baseline(samples))
    
    # CTM
    print("Running CTM standalone...")
    config = CONFIGS[args.config]
    # Override sequence length if specified
    if args.seq_length != config.parity_sequence_length:
        config = CTMParityConfig(
            d_model=config.d_model,
            d_input=config.d_input,
            heads=config.heads,
            n_synch_out=config.n_synch_out,
            n_synch_action=config.n_synch_action,
            iterations=config.iterations,
            memory_length=config.memory_length,
            parity_sequence_length=args.seq_length,
            n_random_pairing_self=config.n_random_pairing_self,
        )
    
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Try default location
        default_checkpoint = INTEGRATION_ROOT / "checkpoints" / "parity_ctm_best.pt"
        if default_checkpoint.exists():
            checkpoint = str(default_checkpoint)
            print(f"Using checkpoint: {checkpoint}")
    
    results.append(benchmark_ctm_standalone(samples, config, checkpoint))
    
    # Print results
    print_results(results)
    
    # Save if requested
    if args.output:
        output_data = [asdict(r) for r in results]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

