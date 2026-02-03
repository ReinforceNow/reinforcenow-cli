# Tutorial: Writing Reward Functions

Learn how to write reward functions for ReinforceNow RL training.

## Quick Start

```bash
# Test the reward function
rnow test -n 3 --verbose

# Once working, train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Training config |
| `train.jsonl` | Sample math problems |
| `rewards.py` | Example reward function using math-verify |
| `requirements.txt` | Dependencies (math-verify) |

## How Rewards Work

1. Model generates a response
2. Each reward function in `rewards` list is called
3. Returns float 0.0-1.0 (higher = better)
4. Total reward = product of all rewards

## Reward Function Signature

```python
from rnow.core import reward, RewardArgs

@reward
def my_reward(args: RewardArgs, messages: list) -> float:
    # args.metadata - data from train.jsonl entry
    # messages - full conversation including assistant response
    response = messages[-1]["content"]
    return 1.0 if "correct" in response else 0.0
```

## Using math-verify

```python
from math_verify import parse, verify
from rnow.core import reward, RewardArgs

@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    gold = parse(args.metadata["answer"])
    pred = parse(messages[-1]["content"])
    return 1.0 if pred and verify(gold, pred) else 0.0
```
