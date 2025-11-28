# NanoCTM: Continuous Thought Machines as Thinking Coprocessors for Language Models

**Authors:** Geby Jaff (UC Berkeley) & Claude 4.5 Opus (Anthropic)

This repository contains the implementation and experiments for integrating Continuous Thought Machines (CTMs) with Transformer language models, enabling iterative "thinking" in latent space.

## Overview

NanoCTM augments Transformer-based language models with CTM refinement heads that perform iterative refinement of hidden representations. Unlike standard Transformers that produce outputs in a single forward pass, CTMs introduce "thinking time" through multiple internal iterations, enabling latent chain-of-thought reasoning.

**Key Result:** On a multi-parity reasoning task, a base Transformer achieves 0% exact-match accuracy, while the same model augmented with a CTM head achieves **58.9%** exact-match accuracy.

## Repository Structure

```
├── integration/
│   ├── adapters/          # CTMHead adapter for Phase 2
│   ├── configs/           # CTM configuration files
│   ├── experiments/      # Main experiments and ablation studies
│   ├── paper/             # LaTeX paper and figures
│   ├── tools/             # Phase 1: CTM as external tool
│   └── train_*.py         # Training scripts
├── continuous-thought-machines/  # CTM implementation (submodule)
└── nanochat/              # Nanochat Transformer (submodule)
```

## Quick Start

### Installation

```bash
# Clone repository (with submodules)
git clone --recurse-submodules https://github.com/spicylemonade/NanoCTM.git
cd NanoCTM

# Install dependencies
pip install torch numpy matplotlib tqdm
```

### Running Experiments

**Phase 2: Unified Nanochat+CTM**
```bash
cd integration
python experiments/unified_experiment.py
```

**Ablation Study: CTM Iterations**
```bash
python experiments/ablation_study.py
```

**Phase 1: CTM as Tool**
```bash
python demo_ctm_tool.py
python chat_with_ctm.py --demo
```

## Key Components

- **`adapters/ctm_head.py`**: CTMHead module that refines Transformer hidden states
- **`experiments/unified_experiment.py`**: Main experiment script with full evaluation
- **`experiments/ablation_study.py`**: Ablation on CTM iteration count
- **`paper/ctm_coprocessor.pdf`**: Complete research paper

## Results

| Model | Bit Accuracy | Sequence Accuracy |
|-------|-------------|-------------------|
| Base Transformer | 43.4% | 0.0% |
| **With CTM** | **91.1%** | **58.9%** |

Performance scales with CTM iterations, peaking at 20 ticks.

## Paper

See `integration/paper/ctm_coprocessor.pdf` for the complete research paper.

## Citation

If you use this work, please cite:

```bibtex
@article{jaff2025,
  title = {Continuous Thought Machines as Thinking Coprocessors for Language Models},
  author = {Geby Jaff and Claude 4.5 Opus},
  year = {2025},
  url = {https://archivara.org/paper/5f25ef52-4487-4c22-af0e-dd1ad3e567d3},
  abstract = {Large Language Models (LLMs) excel at pattern matching but often struggle with tasks requiring multi-step algorithmic reasoning. We explore augmenting Transformer-based language models with \textbf{Continuous Thought Machines (CTMs)}---recurrent neural modules that perform iterative refinement of hidden representations. Unlike standard Transformers that produce outputs in a single forward pass, CTMs introduce ``thinking time'' through multiple internal iterations. We demonstrate this approach on...}
}
```

## License

See individual submodules for their respective licenses.

## Acknowledgments

- CTM architecture from [Sakana AI](https://sakana.ai/ctm/)
- Nanochat Transformer implementation
- Built with PyTorch

