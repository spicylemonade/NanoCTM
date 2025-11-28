"""
Nanochat-CTM Bridge - Integrates CTM tools into nanochat's generation loop.

This implements the tool-call protocol where nanochat can emit special tokens
like [CTM_PARITY] to invoke CTM reasoning, and receive [CTM_RESULT] back.
"""
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass

import torch

# Add paths
INTEGRATION_ROOT = Path(__file__).parent.parent
NANOCHAT_ROOT = INTEGRATION_ROOT.parent / "nanochat"
CTM_ROOT = INTEGRATION_ROOT.parent / "continuous-thought-machines"

sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(NANOCHAT_ROOT))
sys.path.insert(0, str(CTM_ROOT))

from configs.ctm_config import IntegrationConfig
from tools.ctm_tool import CTMToolRouter, CTMParityTool, CTMParityConfig


@dataclass
class ToolCallResult:
    """Result of a tool call"""
    tool_name: str
    input_text: str
    output_text: str
    success: bool
    metadata: Dict[str, Any]


class NanochatCTMBridge:
    """
    Bridge between nanochat and CTM tools.
    
    Usage modes:
    1. Post-generation interception: Check generated text for tool calls, execute, append result
    2. Token-by-token interception: Monitor generation for tool tokens, pause and execute
    3. Prompt injection: Add tool usage examples to system prompt
    """
    
    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        auto_register_tools: bool = True
    ):
        self.config = config or IntegrationConfig()
        self.router = CTMToolRouter(self.config)
        self.tool_call_history: List[ToolCallResult] = []
        
        if auto_register_tools:
            self._register_default_tools()
            
    def _register_default_tools(self):
        """Register available CTM tools"""
        # Parity tool (lightweight, fast)
        parity_config = CTMParityConfig(
            d_model=256,
            iterations=25,
            memory_length=8,
            n_synch_out=64,
            n_synch_action=64,
            n_random_pairing_self=8
        )
        parity_tool = CTMParityTool(config=parity_config)
        self.router.register_tool("PARITY", parity_tool)
        
    def register_tool(self, name: str, tool: Any):
        """Register a custom CTM tool"""
        self.router.register_tool(name, tool)
        
    def get_system_prompt(self) -> str:
        """
        Get system prompt that teaches nanochat about CTM tools.
        Add this to the conversation for tool-augmented generation.
        """
        return """You have access to specialized computational tools for tasks that require iterative reasoning.

Available tools:
1. [CTM_PARITY] - Computes cumulative parity of a bit sequence
   Usage: [CTM_PARITY] 1 -1 1 1 -1
   The tool will respond with [CTM_RESULT] containing the parities

When you encounter a task requiring:
- Parity computation on bit sequences
- Complex algorithmic reasoning that benefits from iterative thinking

You should emit the appropriate tool call and wait for the result.

Example:
User: What is the cumulative parity of the sequence 1, -1, 1, 1?
You: Let me compute this using the parity tool.
[CTM_PARITY] 1 -1 1 1
[CTM_RESULT] parities=1 -1 -1 -1 confidence=0.98 tick=15
The cumulative parities are: 1, -1, -1, -1"""

    def detect_tool_call(self, text: str) -> Optional[tuple]:
        """
        Detect if text contains a tool call.
        Returns (tool_name, args) or None.
        """
        return self.router.parse_tool_call(text)
        
    def execute_tool_call(self, text: str) -> Optional[ToolCallResult]:
        """
        Execute a tool call found in text.
        Returns ToolCallResult or None if no tool call found.
        """
        parsed = self.detect_tool_call(text)
        if parsed is None:
            return None
            
        tool_name, args_text = parsed
        output = self.router.execute(tool_name, args_text)
        
        success = "[CTM_ERROR]" not in output
        result = ToolCallResult(
            tool_name=tool_name,
            input_text=args_text,
            output_text=output,
            success=success,
            metadata={}
        )
        
        self.tool_call_history.append(result)
        return result
        
    def process_generation(
        self,
        generated_text: str,
        generate_fn: Callable[[List[int]], Generator],
        tokenizer: Any,
        conversation_tokens: List[int],
        max_tool_iterations: int = 3
    ) -> tuple:
        """
        Process a generation, executing tool calls and continuing generation.
        
        Args:
            generated_text: The text generated so far
            generate_fn: Function to generate more tokens
            tokenizer: Tokenizer for encoding/decoding
            conversation_tokens: Current conversation token history
            max_tool_iterations: Max tool calls per response
            
        Returns:
            (final_text, updated_conversation_tokens, tool_calls)
        """
        tool_calls = []
        current_text = generated_text
        
        for _ in range(max_tool_iterations):
            # Check for tool call
            result = self.execute_tool_call(current_text)
            if result is None:
                break
                
            tool_calls.append(result)
            
            # Append tool result to conversation
            tool_response = f"\n{result.output_text}\n"
            tool_tokens = tokenizer.encode(tool_response)
            conversation_tokens.extend(tool_tokens)
            
            # Continue generation after tool result
            new_tokens = []
            for token_column, _ in generate_fn(conversation_tokens):
                token = token_column[0]
                new_tokens.append(token)
                
            current_text += tool_response + tokenizer.decode(new_tokens)
            conversation_tokens.extend(new_tokens)
            
        return current_text, conversation_tokens, tool_calls


class StreamingToolInterceptor:
    """
    Intercepts streaming token generation to detect and execute tool calls.
    
    This wraps around nanochat's token-by-token generation to:
    1. Accumulate tokens until a tool call is detected
    2. Execute the tool call
    3. Inject the result back into the stream
    """
    
    def __init__(self, bridge: NanochatCTMBridge, tokenizer: Any):
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.buffer = []
        self.tool_pattern = re.compile(r'\[CTM_(\w+)\]\s*([^\n\[]*)')
        
    def process_token(self, token: int) -> tuple:
        """
        Process a single generated token.
        
        Returns:
            (should_yield, tokens_to_yield, tool_executed)
        """
        self.buffer.append(token)
        buffer_text = self.tokenizer.decode(self.buffer)
        
        # Check if we have a complete tool call
        match = self.tool_pattern.search(buffer_text)
        if match:
            tool_name = match.group(1)
            args_text = match.group(2).strip()
            
            # Check if args look complete (has some content, no unclosed brackets)
            if args_text and not buffer_text.rstrip().endswith('['):
                # Execute tool
                result = self.bridge.execute_tool_call(buffer_text)
                if result:
                    # Clear buffer and return tool result
                    self.buffer = []
                    result_tokens = self.tokenizer.encode(f"\n{result.output_text}\n")
                    return True, result_tokens, True
                    
        # Normal passthrough - yield oldest buffered tokens when buffer gets long
        if len(self.buffer) > 20:  # Keep some buffer for pattern matching
            tokens_to_yield = self.buffer[:10]
            self.buffer = self.buffer[10:]
            return True, tokens_to_yield, False
            
        return False, [], False
        
    def flush(self) -> List[int]:
        """Flush remaining buffer"""
        tokens = self.buffer
        self.buffer = []
        return tokens


def create_augmented_chat_loop(
    model,
    tokenizer,
    bridge: NanochatCTMBridge,
    device: str = "cuda"
):
    """
    Create an augmented chat loop with CTM tool support.
    
    This is a drop-in replacement for nanochat's chat loop that adds
    CTM tool execution.
    """
    from nanochat.engine import Engine
    
    engine = Engine(model, tokenizer)
    
    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    
    # Add system prompt with tool instructions
    system_prompt = bridge.get_system_prompt()
    conversation_tokens = [bos]
    system_tokens = tokenizer.encode(system_prompt)
    conversation_tokens.extend(system_tokens)
    
    print("\nNanoChat + CTM Tools")
    print("-" * 50)
    print("Available tools: PARITY")
    print("Type 'quit' to exit, 'clear' to reset")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input.lower() == 'clear':
            conversation_tokens = [bos] + system_tokens
            print("Conversation cleared.")
            continue
        if not user_input:
            continue
            
        # Add user message
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        
        # Generate with tool interception
        print("\nAssistant: ", end="", flush=True)
        
        response_text = ""
        response_tokens = []
        interceptor = StreamingToolInterceptor(bridge, tokenizer)
        
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 256,
            "temperature": 0.6,
            "top_k": 50,
        }
        
        for token_column, _ in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)
            
            should_yield, tokens, tool_executed = interceptor.process_token(token)
            
            if tool_executed:
                # Tool was executed, inject result
                response_tokens.extend(tokens)
                result_text = tokenizer.decode(tokens)
                print(f"\n[Tool executed]{result_text}", end="", flush=True)
            elif should_yield:
                text = tokenizer.decode(tokens)
                print(text, end="", flush=True)
                
        # Flush remaining
        remaining = interceptor.flush()
        if remaining:
            print(tokenizer.decode(remaining), end="", flush=True)
            response_tokens.extend(remaining)
            
        print()
        
        # Ensure assistant end token
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)
        
        
if __name__ == "__main__":
    # Test the bridge standalone
    print("Testing NanochatCTMBridge...")
    
    bridge = NanochatCTMBridge()
    
    # Test tool detection
    test_texts = [
        "Let me compute this: [CTM_PARITY] 1 -1 1 1",
        "No tool call here",
        "[CTM_PARITY] 1 1 1 -1 -1 1 1 -1",
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        result = bridge.execute_tool_call(text)
        if result:
            print(f"Tool: {result.tool_name}")
            print(f"Output: {result.output_text}")
        else:
            print("No tool call detected")

