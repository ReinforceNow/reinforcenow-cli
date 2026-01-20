---
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
tags:
  - reinforcement-learning
  - RLHF
  - ReinforceNow
  - math
  - reasoning
language:
  - en
size_categories:
  - 1K<n<10K
---

# RL-Single: Math Reasoning Dataset for ReinforceNow

A dataset compatible with the **[ReinforceNow Platform](https://reinforcenow.ai)** for RLHF (Reinforcement Learning from Human Feedback) training of mathematical reasoning models.

## Overview

This dataset contains mathematical reasoning problems sourced from AoPS (Art of Problem Solving) forums, formatted for use with the ReinforceNow CLI. It is designed for training language models to solve complex math problems using reinforcement learning.

## Quick Start

```bash
# Install the ReinforceNow CLI
pip install rnow

# Initialize a project with this template
rnow init -t rl-single -n "My Math Model"

# Start training
rnow run
```

## Dataset Structure

```
├── config.yml          # Training configuration
├── rewards.py          # Reward functions for RL training
├── train.jsonl         # Training data (math problems)
└── requirements.txt    # Python dependencies
```

## Reward Function

The dataset uses `math-verify` for accurate mathematical answer verification:

```python
from math_verify import LatexExtractionConfig, parse, verify
from rnow.core import RewardArgs, get_response, reward

@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    gold = parse(args.metadata["expected_answer"])
    pred = parse(
        get_response(messages),
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
```

This reward function:
- Extracts the expected answer from metadata
- Parses the model's response looking for `\boxed{}` formatted answers
- Uses symbolic math verification to check correctness
- Returns 1.0 for correct answers, 0.0 otherwise

## Dependencies

```txt
math-verify==0.5.0
```

## Data Format

Each entry in `train.jsonl` follows this structure:

```json
{
  "messages": [
    {"role": "user", "content": "Solve the following math problem..."}
  ],
  "rewards": ["accuracy"],
  "metadata": {"expected_answer": "\\(\\frac{4}{5}\\)"}
}
```

## Training Configuration

Default configuration (`config.yml`):

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-8B` |
| Algorithm | PPO with GRPO advantage estimator |
| Batch Size | 16 |
| Group Size | 8 |
| Max Tokens | 16000 |
| Learning Rate | 0.0001 |
| Epochs | 6 |
| QLoRA Rank | 32 |

## Example Problems

The dataset includes challenging mathematical problems:

- **Optimization**: Find minimum values subject to constraints
- **Geometry**: Triangle problems, angle calculations
- **Algebra**: Polynomial factorization, functional equations
- **Inequalities**: Finding optimal constants

## Links

- [ReinforceNow Platform](https://reinforcenow.ai)
- [ReinforceNow CLI Documentation](https://reinforcenow.ai/docs)
- [math-verify Package](https://pypi.org/project/math-verify/)
- Inspired by [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning)

## License

This dataset is released under the **CC-BY-4.0** license.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{reinforcenow2025,
  title={RL-Single: Math Reasoning Dataset for ReinforceNow},
  author={ReinforceNow Team},
  year={2025},
  publisher={Hugging Face}
}
```
