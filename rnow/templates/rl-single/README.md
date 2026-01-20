---
license: cc-by-4.0
tags:
  - reinforcement-learning
  - RLHF
  - ReinforceNow
  - math
---

# RL-Single: Test Dataset for ReinforceNow

This is a **test dataset** compatible with the [ReinforceNow Platform](https://reinforcenow.ai) containing only **92 sample entries** for demonstration and testing purposes. It is designed to help users get started with RLHF training for mathematical reasoning models.

> **Note:** This is not a production dataset. It contains a small subset of math problems for testing the ReinforceNow CLI and verifying your training pipeline works correctly before scaling up.

## Data Format

Each entry in `train.jsonl` follows this structure:

```json
{"messages": [{"role": "user", "content": "Problem..."}], "rewards": ["accuracy"], "metadata": {"expected_answer": "$\\frac{1}{2}$"}}
```

**Important:** All `expected_answer` values use proper math delimiters (`$...$` or `\(...\)`) for compatibility with `math-verify`. Plain numbers like `42` work as-is.

## Reward Function

The dataset uses the `math-verify` library for accurate mathematical answer verification:

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

## Dependencies

```
math-verify==0.5.0
```

## Links

- [ReinforceNow Platform](https://reinforcenow.ai)
- [ReinforceNow CLI](https://pypi.org/project/rnow/)
