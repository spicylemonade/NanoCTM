#!/usr/bin/env python3
"""
Chat with nanochat + CTM Integration

This script provides an interactive chat that can use CTM as a thinking coprocessor.
When the model outputs [CTM_PARITY], the CTM tool is automatically invoked.

Usage:
    python chat_with_ctm.py              # Interactive mode
    python chat_with_ctm.py --demo       # Demo mode with example prompts
"""
import argparse
import sys
from pathlib import Path

# Setup paths
INTEGRATION_ROOT = Path(__file__).parent
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT.parent / "continuous-thought-machines"))
sys.path.insert(0, str(INTEGRATION_ROOT.parent / "nanochat"))

from tools.ctm_tool import CTMParityTool, CTMToolRouter
from configs.ctm_config import CONFIGS


def create_ctm_tools():
    """Initialize CTM tools"""
    print("Loading CTM Parity Tool...")
    config = CONFIGS['parity_quick']
    tool = CTMParityTool(
        checkpoint_path=str(INTEGRATION_ROOT / 'checkpoints/parity_ctm_final.pt'),
        config=config
    )
    
    router = CTMToolRouter()
    router.register_tool('PARITY', tool)
    
    return router, tool


def process_with_ctm(text: str, router: CTMToolRouter) -> str:
    """
    Process text and execute any CTM tool calls.
    Returns the text with tool results injected.
    """
    parsed = router.parse_tool_call(text)
    if parsed:
        tool_name, args = parsed
        result = router.execute(tool_name, args)
        
        # Find the tool call and inject result after it
        marker = f"[CTM_{tool_name}]"
        idx = text.find(marker)
        if idx >= 0:
            # Find end of line after marker
            line_end = text.find('\n', idx)
            if line_end == -1:
                line_end = len(text)
            
            # Insert result
            text = text[:line_end] + f"\n{result}" + text[line_end:]
    
    return text


def demo_mode(router, tool):
    """Run demo with example problems"""
    print("\n" + "=" * 60)
    print("  CTM-Enhanced Chat Demo")
    print("=" * 60)
    
    examples = [
        {
            "user": "Can you compute the cumulative parity of: 1, -1, 1, 1, -1, -1?",
            "simulated_response": """I'll compute the cumulative parity for you. 

The cumulative parity at each position is the product of all elements up to that point.

[CTM_PARITY] 1 -1 1 1 -1 -1 1 1 1 1 1 1 1 1 1 1

Based on this computation, here are the results:
- Position 1: 1 (just the first element)
- Position 2: -1 (1 × -1 = -1)
- Position 3: -1 (-1 × 1 = -1)  
- Position 4: -1 (-1 × 1 = -1)
- Position 5: 1 (-1 × -1 = 1)
- Position 6: -1 (1 × -1 = -1)"""
        },
        {
            "user": "What about a longer sequence: 1, 1, -1, -1, 1, -1, 1, 1, -1, -1?",
            "simulated_response": """Let me compute the cumulative parity for this 10-element sequence:

[CTM_PARITY] 1 1 -1 -1 1 -1 1 1 -1 -1 1 1 1 1 1 1

The CTM has computed the running product (parity) at each position."""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"\n👤 User: {example['user']}")
        
        # Process with CTM
        response = process_with_ctm(example['simulated_response'], router)
        
        print(f"\n🤖 Assistant:")
        print(response)
        print()
    
    # Also show accuracy stats
    print("\n" + "=" * 60)
    print("  CTM Performance Summary")
    print("=" * 60)
    
    import random
    random.seed(123)
    
    correct = 0
    total = 0
    for _ in range(50):
        bits = [random.choice([-1, 1]) for _ in range(16)]
        gt = tool.compute_ground_truth(bits)
        pred = tool(bits)['parities']
        correct += sum(1 for p, g in zip(pred, gt) if p == g)
        total += 16
    
    print(f"\n  Per-bit accuracy: {100*correct/total:.1f}%")
    print(f"  Improvement over random: +{100*correct/total - 50:.1f}%")
    print()


def interactive_mode(router):
    """Interactive chat mode"""
    print("\n" + "=" * 60)
    print("  CTM-Enhanced Interactive Chat")
    print("=" * 60)
    print("\nThis is a simulation showing how CTM would integrate with nanochat.")
    print("Type messages and the assistant will use CTM when needed.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'parity <bits>' - Directly compute parity (e.g., 'parity 1 -1 1 1')")
    print()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle direct parity command
        if user_input.lower().startswith('parity '):
            bits_str = user_input[7:].strip()
            text = f"[CTM_PARITY] {bits_str}"
            result = process_with_ctm(text, router)
            print(f"\n🤖 CTM: {result}")
            continue
        
        # Simulate assistant response with potential CTM call
        # In real integration, this would be nanochat's output
        if any(word in user_input.lower() for word in ['parity', 'cumulative', 'product', 'bits']):
            # Extract numbers if present
            import re
            numbers = re.findall(r'-?\d+', user_input)
            if numbers and len(numbers) >= 4:
                bits = ' '.join(numbers[:16])  # Take up to 16 bits
                response = f"""I'll compute the cumulative parity using CTM:

[CTM_PARITY] {bits} {'1 ' * max(0, 16 - len(numbers))}

The CTM has computed the running product at each position.
Each value shows whether there's an even (1) or odd (-1) number of -1s up to that point."""
                response = process_with_ctm(response, router)
                print(f"\n🤖 Assistant:\n{response}")
            else:
                print("\n🤖 Assistant: Please provide a sequence of bits (1 or -1) for me to compute the parity.")
        else:
            print("\n🤖 Assistant: I'm a demo assistant with CTM integration. Ask me to compute cumulative parity of a bit sequence!")
            print("   Example: 'What is the cumulative parity of 1, -1, 1, 1, -1, -1?'")


def main():
    parser = argparse.ArgumentParser(description='Chat with nanochat + CTM')
    parser.add_argument('--demo', action='store_true', help='Run demo mode with examples')
    args = parser.parse_args()
    
    print("\n🧠 CTM + nanochat Integration")
    print("   Continuous Thought Machine as Thinking Coprocessor\n")
    
    # Initialize CTM
    router, tool = create_ctm_tools()
    
    if args.demo:
        demo_mode(router, tool)
    else:
        interactive_mode(router)


if __name__ == "__main__":
    main()

