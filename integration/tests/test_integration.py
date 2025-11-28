#!/usr/bin/env python3
"""
Test harness for CTM-Nanochat integration.

Tests:
1. CTM tool standalone inference
2. Tool call parsing and execution
3. Comparison metrics between nanochat alone vs with CTM tools

Usage:
    cd /home/spicylemon/Desktop/ctm_nano/integration
    python -m pytest tests/test_integration.py -v
    # or
    python tests/test_integration.py
"""
import sys
from pathlib import Path
import time

# Add paths
INTEGRATION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(INTEGRATION_ROOT))

import torch
import numpy as np


def test_parity_tool_initialization():
    """Test CTM parity tool can be initialized"""
    from tools.ctm_tool import CTMParityTool
    from configs.ctm_config import CTMParityConfig
    
    config = CTMParityConfig(d_model=128, iterations=10, memory_length=4)
    tool = CTMParityTool(config=config)
    
    assert tool.model is not None
    assert tool.device in ['cuda', 'mps', 'cpu']
    print(f"✓ Parity tool initialized on {tool.device}")


def test_parity_ground_truth():
    """Test parity ground truth computation"""
    from tools.ctm_tool import CTMParityTool
    from configs.ctm_config import CTMParityConfig
    
    config = CTMParityConfig(d_model=128, iterations=10, memory_length=4)
    tool = CTMParityTool(config=config)
    
    # Test cases
    test_cases = [
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([1, -1, 1, -1], [1, -1, -1, 1]),
        ([-1, -1, -1, -1], [-1, 1, -1, 1]),
        ([1, -1, -1, 1], [1, -1, 1, 1]),
    ]
    
    for bits, expected in test_cases:
        result = tool.compute_ground_truth(bits)
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Parity ground truth computation correct")


def test_parity_tool_inference():
    """Test CTM parity tool can run inference"""
    from tools.ctm_tool import CTMParityTool
    from configs.ctm_config import CTMParityConfig
    
    config = CTMParityConfig(d_model=128, iterations=10, memory_length=4)
    tool = CTMParityTool(config=config)
    
    bits = [1, -1, 1, 1, -1, 1, -1, -1]
    result = tool(bits)
    
    assert 'parities' in result
    assert 'confidence' in result
    assert 'tick_used' in result
    assert len(result['parities']) == config.parity_sequence_length
    assert 0 <= result['confidence'] <= 1
    assert 0 <= result['tick_used'] < config.iterations
    
    print(f"✓ Parity tool inference works")
    print(f"  Input: {bits}")
    print(f"  Output parities (first 8): {result['parities'][:8]}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Tick used: {result['tick_used']}")


def test_tool_router():
    """Test tool call parsing and routing"""
    from tools.ctm_tool import CTMToolRouter, CTMParityTool
    from configs.ctm_config import IntegrationConfig, CTMParityConfig
    
    config = IntegrationConfig()
    router = CTMToolRouter(config)
    
    # Register parity tool
    parity_config = CTMParityConfig(d_model=128, iterations=10, memory_length=4)
    parity_tool = CTMParityTool(config=parity_config)
    router.register_tool("PARITY", parity_tool)
    
    # Test parsing
    test_cases = [
        ("[CTM_PARITY] 1 -1 1 1", ("PARITY", "1 -1 1 1")),
        ("Let me compute: [CTM_PARITY] 1 1 -1", ("PARITY", "1 1 -1")),
        ("No tool call here", None),
    ]
    
    for text, expected in test_cases:
        result = router.parse_tool_call(text)
        assert result == expected, f"For '{text}': expected {expected}, got {result}"
    
    # Test execution
    result = router.execute("PARITY", "1 -1 1 1")
    assert "[CTM_RESULT]" in result
    assert "parities=" in result
    assert "confidence=" in result
    
    print("✓ Tool router parsing and execution works")


def test_bridge():
    """Test the nanochat-CTM bridge"""
    from tools.nanochat_ctm_bridge import NanochatCTMBridge
    
    bridge = NanochatCTMBridge(auto_register_tools=True)
    
    # Test tool detection
    result = bridge.execute_tool_call("[CTM_PARITY] 1 -1 1 1 -1 1 1 -1")
    
    assert result is not None
    assert result.tool_name == "PARITY"
    assert result.success
    assert "[CTM_RESULT]" in result.output_text
    
    # Test no tool call
    result = bridge.execute_tool_call("Just regular text")
    assert result is None
    
    print("✓ Nanochat-CTM bridge works")


def test_inference_speed():
    """Benchmark CTM inference speed"""
    from tools.ctm_tool import CTMParityTool
    from configs.ctm_config import CTMParityConfig, CONFIGS
    
    print("\nInference Speed Benchmark:")
    print("-" * 50)
    
    for name, config in CONFIGS.items():
        if not name.startswith('parity'):
            continue
            
        tool = CTMParityTool(config=config)
        
        # Warmup
        for _ in range(3):
            tool([1, -1, 1, 1])
        
        # Benchmark
        n_trials = 20
        bits = [1 if np.random.random() > 0.5 else -1 for _ in range(config.parity_sequence_length)]
        
        start = time.time()
        for _ in range(n_trials):
            tool(bits)
        elapsed = time.time() - start
        
        ms_per_call = (elapsed / n_trials) * 1000
        print(f"{name}: {ms_per_call:.2f}ms per call")
        print(f"  d_model={config.d_model}, iterations={config.iterations}")


def test_accuracy_with_random_weights():
    """Test accuracy with random weights (baseline)"""
    from tools.ctm_tool import CTMParityTool
    from configs.ctm_config import CTMParityConfig
    
    config = CTMParityConfig(d_model=128, iterations=10, memory_length=4)
    tool = CTMParityTool(config=config)
    
    n_samples = 100
    correct = 0
    total = 0
    
    for _ in range(n_samples):
        bits = [1 if np.random.random() > 0.5 else -1 for _ in range(8)]
        gt = tool.compute_ground_truth(bits)
        result = tool(bits)
        
        for pred, truth in zip(result['parities'][:8], gt):
            if pred == truth:
                correct += 1
            total += 1
    
    acc = correct / total
    print(f"\nRandom weight accuracy: {acc:.2%} (expected ~50%)")
    # With random weights, should be around chance
    assert 0.3 < acc < 0.7, f"Accuracy {acc} seems off for random weights"
    print("✓ Random weight baseline reasonable")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CTM-Nanochat Integration Tests")
    print("=" * 60)
    
    tests = [
        test_parity_tool_initialization,
        test_parity_ground_truth,
        test_parity_tool_inference,
        test_tool_router,
        test_bridge,
        test_accuracy_with_random_weights,
        test_inference_speed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

