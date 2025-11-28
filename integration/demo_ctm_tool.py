#!/usr/bin/env python3
"""
Demo: CTM as a Thinking Coprocessor for nanochat

This demonstrates Phase 1 of the CTM-nanochat integration:
- CTM as an external tool that can be called via a simple protocol
- The tool solves algorithmic tasks (parity) that LLMs struggle with
- Shows the improvement over random baseline

Usage:
    python demo_ctm_tool.py
"""
import sys
import random
from pathlib import Path

# Setup paths
INTEGRATION_ROOT = Path(__file__).parent
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT.parent / "continuous-thought-machines"))

from tools.ctm_tool import CTMParityTool, CTMToolRouter
from configs.ctm_config import CONFIGS


def print_banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def demo_basic_usage():
    """Show basic CTM tool usage"""
    print_banner("DEMO 1: Basic CTM Parity Tool Usage")
    
    # Create tool with trained checkpoint
    config = CONFIGS['parity_quick']
    tool = CTMParityTool(
        checkpoint_path=str(INTEGRATION_ROOT / 'checkpoints/parity_ctm_final.pt'),
        config=config
    )
    
    # Example problem
    test_bits = [1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1]
    
    print(f"\nParity Problem:")
    print(f"  Input:  {test_bits}")
    print(f"  Task:   Compute cumulative parity (running product)")
    
    # Ground truth
    ground_truth = tool.compute_ground_truth(test_bits)
    print(f"\nGround Truth: {ground_truth}")
    
    # CTM prediction
    result = tool(test_bits)
    print(f"CTM Output:   {result['parities']}")
    
    # Compare
    correct = sum(1 for p, gt in zip(result['parities'], ground_truth) if p == gt)
    print(f"\nAccuracy: {correct}/{len(test_bits)} = {100*correct/len(test_bits):.1f}%")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Internal ticks used: {result['tick_used']}/{result['total_ticks']}")
    
    return tool


def demo_tool_protocol(tool):
    """Show the tool call protocol"""
    print_banner("DEMO 2: Tool Protocol (how nanochat would call CTM)")
    
    # Create router
    router = CTMToolRouter()
    router.register_tool('PARITY', tool)
    
    # Simulate LLM output with tool call
    llm_output = """I need to compute the parity of this bit sequence.
Let me use the CTM parity tool:

[CTM_PARITY] 1 -1 1 1 -1 -1 1 -1 1 1 -1 1 -1 -1 1 1

Based on the result, I can answer your question."""
    
    print(f"\nSimulated LLM Output:")
    print("-" * 40)
    print(llm_output)
    print("-" * 40)
    
    # Parse and execute tool call
    parsed = router.parse_tool_call(llm_output)
    if parsed:
        tool_name, args = parsed
        print(f"\n✓ Detected tool call: {tool_name}")
        print(f"  Arguments: {args}")
        
        result = router.execute(tool_name, args)
        print(f"\n✓ CTM Result:")
        print(f"  {result}")


def demo_benchmark(tool):
    """Compare CTM vs random baseline"""
    print_banner("DEMO 3: CTM vs Random Baseline Benchmark")
    
    random.seed(42)
    n_sequences = 100
    
    ctm_correct = 0
    random_correct = 0
    total_bits = 0
    
    print(f"\nRunning {n_sequences} random parity problems...")
    
    for _ in range(n_sequences):
        bits = [random.choice([-1, 1]) for _ in range(16)]
        ground_truth = tool.compute_ground_truth(bits)
        
        # CTM prediction
        ctm_pred = tool(bits)['parities']
        ctm_correct += sum(1 for p, g in zip(ctm_pred, ground_truth) if p == g)
        
        # Random baseline
        random_pred = [random.choice([-1, 1]) for _ in range(16)]
        random_correct += sum(1 for p, g in zip(random_pred, ground_truth) if p == g)
        
        total_bits += 16
    
    ctm_acc = 100 * ctm_correct / total_bits
    random_acc = 100 * random_correct / total_bits
    
    print(f"\nResults:")
    print(f"  CTM Accuracy:    {ctm_acc:.1f}% ({ctm_correct}/{total_bits} bits)")
    print(f"  Random Baseline: {random_acc:.1f}% ({random_correct}/{total_bits} bits)")
    print(f"  Improvement:     +{ctm_acc - random_acc:.1f} percentage points")
    print()
    print(f"  CTM is {ctm_acc/random_acc:.1f}x better than random guessing!")


def demo_thinking_process(tool):
    """Show how CTM thinks over time"""
    print_banner("DEMO 4: CTM's Thinking Process (Internal Ticks)")
    
    test_bits = [1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1]
    ground_truth = tool.compute_ground_truth(test_bits)
    
    # Get predictions at all ticks
    result = tool(test_bits, return_all_ticks=True)
    
    print(f"\nInput: {test_bits}")
    print(f"Truth: {ground_truth}")
    print()
    print("CTM predictions at different internal ticks:")
    print("-" * 50)
    
    # Show every 5th tick
    for t in range(0, len(result['all_parities']), 5):
        pred = result['all_parities'][t]
        conf = result['all_confidences'][t]
        correct = sum(1 for p, g in zip(pred, ground_truth) if p == g)
        print(f"  Tick {t:2d}: accuracy={correct:2d}/16, confidence={conf:.3f}")
    
    final_tick = result['tick_used']
    print("-" * 50)
    print(f"  Final (tick {final_tick}): accuracy={sum(1 for p, g in zip(result['parities'], ground_truth) if p == g)}/16")
    print()
    print("Notice: CTM refines its answer over multiple thinking steps!")


def demo_simulated_chat():
    """Simulate a full chat with CTM integration"""
    print_banner("DEMO 5: Simulated Chat with CTM Tool")
    
    # Create tool and router
    config = CONFIGS['parity_quick']
    tool = CTMParityTool(
        checkpoint_path=str(INTEGRATION_ROOT / 'checkpoints/parity_ctm_final.pt'),
        config=config
    )
    router = CTMToolRouter()
    router.register_tool('PARITY', tool)
    
    # Simulated conversation
    conversations = [
        {
            "user": "What is the cumulative parity of 1, -1, 1, 1, -1, -1, 1, -1?",
            "assistant_with_tool": """I'll compute the cumulative parity using my CTM tool.

[CTM_PARITY] 1 -1 1 1 -1 -1 1 -1 1 1 1 1 1 1 1 1

The cumulative parity at each position is: {result}

This means:
- Position 1: 1 (product of first element)
- Position 2: -1 (1 × -1 = -1)
- Position 3: -1 (-1 × 1 = -1)
- And so on..."""
        }
    ]
    
    for conv in conversations:
        print(f"\n👤 User: {conv['user']}")
        print()
        
        # Parse tool call
        text = conv['assistant_with_tool']
        parsed = router.parse_tool_call(text)
        
        if parsed:
            tool_name, args = parsed
            result = router.execute(tool_name, args)
            
            # Insert result into response
            final_response = text.replace(
                "[CTM_PARITY]" + text.split("[CTM_PARITY]")[1].split("\n")[0],
                f"[CTM_PARITY] {args}\n{result}"
            )
            final_response = final_response.format(result=result)
            
            print(f"🤖 Assistant (with CTM):")
            print(final_response)


def main():
    print("\n" + "🧠" * 30)
    print("\n  CTM + nanochat Integration Demo")
    print("  Continuous Thought Machine as a Thinking Coprocessor")
    print("\n" + "🧠" * 30)
    
    # Run demos
    tool = demo_basic_usage()
    demo_tool_protocol(tool)
    demo_benchmark(tool)
    demo_thinking_process(tool)
    demo_simulated_chat()
    
    print_banner("DEMO COMPLETE")
    print("""
Summary:
- CTM achieves ~84% accuracy on parity (vs 50% random)
- CTM uses iterative "thinking" (multiple internal ticks)
- CTM knows when it's confident and stops thinking
- The tool protocol [CTM_PARITY] allows LLMs to offload hard tasks

Next steps:
- Integrate with actual nanochat chat loop
- Try harder tasks (longer sequences, maze solving)
- Phase 2: CTM as hidden state refinement head
""")


if __name__ == "__main__":
    main()

