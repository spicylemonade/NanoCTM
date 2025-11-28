# CTM + Nanochat Integration

**Continuous Thought Machines as a Thinking Coprocessor for LLMs**

This integration treats CTM as an external computational tool that nanochat can call for tasks requiring iterative reasoning (like algorithmic tasks that LLMs typically struggle with).

## What is CTM?

CTM (Continuous Thought Machine) is **NOT an LLM**. It's a fundamentally different architecture:

| Feature | LLM (Transformer) | CTM |
|---------|-------------------|-----|
| Core operation | Predict next token | Iterate internal "thought" |
| Time axis | Sequence position | Internal ticks (T iterations) |
| Neurons | Stateless | **Stateful** (keep history) |
| Stopping | Fixed (one pass) | Can stop when **confident** |

### How CTM Thinks

```
Input → Backbone → [ITERATIVE LOOP: 25 ticks] → Output
                          ↓
        For each tick:
        1. Compute "synchronization" between neurons
        2. Query input via attention using sync
        3. Apply "synapses" (U-NET mixing neurons)  
        4. Update neuron state history (memory)
        5. Apply per-neuron MLPs (NLMs) to history
        6. Make prediction & compute certainty
```

## Quick Start

### Run the Demo
```bash
cd integration
python demo_ctm_tool.py
```

### Interactive Chat with CTM
```bash
python chat_with_ctm.py           # Interactive mode
python chat_with_ctm.py --demo    # Demo mode
```

### Direct Tool Usage
```python
from tools.ctm_tool import CTMParityTool
from configs.ctm_config import CONFIGS

tool = CTMParityTool(
    checkpoint_path='checkpoints/parity_ctm_final.pt',
    config=CONFIGS['parity_quick']
)

# Compute parity
bits = [1, -1, 1, 1, -1, 1, -1, -1]
result = tool(bits)
print(result['parities'])  # Cumulative parity at each position
print(result['confidence']) # How certain CTM is
print(result['tick_used'])  # Which internal tick was used
```

## Performance

| Task | CTM Accuracy | Random Baseline | Improvement |
|------|--------------|-----------------|-------------|
| 16-bit Parity | **84%** | 50% | **+34%** |

The CTM can:
- ✅ Iteratively refine answers over multiple internal "thinking" steps
- ✅ Know when it's confident and stop early
- ✅ Handle algorithmic tasks that LLMs struggle with

## Project Structure

```
integration/
├── configs/
│   └── ctm_config.py          # CTM model configurations
├── tools/
│   ├── ctm_tool.py            # CTM parity tool wrapper
│   └── nanochat_ctm_bridge.py # LLM-CTM bridge (future)
├── adapters/
│   └── ctm_head.py            # Phase 2: CTMHead adapter
├── checkpoints/
│   └── parity_ctm_final.pt    # Trained CTM model (84% acc)
├── demo_ctm_tool.py           # Full feature demo
├── chat_with_ctm.py           # Interactive chat with CTM
├── train_parity_ctm.py        # Training script
└── README.md                  # This file
```

## Tool Protocol

The integration uses a simple text-based protocol:

**Input (LLM outputs this):**
```
[CTM_PARITY] 1 -1 1 1 -1 -1 1 -1 1 1 -1 1 -1 -1 1 1
```

**Output (CTM returns this):**
```
[CTM_RESULT] parities=1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 1 1 -1 1 confidence=0.85 tick=18
```

## Integration Phases

### Phase 1: CTM as External Tool ✅ COMPLETE
- CTM runs as a separate model
- LLM calls it via text protocol
- Results injected back into conversation

### Phase 2: CTM as Hidden State Refinement (Future)
- CTM takes LLM's hidden states
- Refines them for specific tasks
- Single forward pass through both

### Phase 3: Hybrid Architecture (Research)
- CTM blocks inside the Transformer
- True architectural fusion

## Training Your Own CTM

```bash
# Quick training (5k iterations, ~30 min)
python train_parity_ctm.py --config parity_quick --training_iterations 5000

# Full training (20k iterations, ~2 hours)  
python train_parity_ctm.py --config parity_notebook --training_iterations 20000
```

## Key Files

- **`tools/ctm_tool.py`**: Main CTM wrapper with encoding/decoding logic
- **`configs/ctm_config.py`**: Model configurations for different tasks
- **`train_parity_ctm.py`**: Training script matching original CTM repo

## References

- [CTM Paper](https://arxiv.org/abs/2505.05522)
- [CTM Interactive Website](https://pub.sakana.ai/ctm/)
- [Original CTM Repo](https://github.com/SakanaAI/continuous-thought-machines)
