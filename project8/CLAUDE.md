# RL Single-Turn Template

Train a model using reinforcement learning with reward functions.

## Quick Start

```bash
# 1. Prepare train.jsonl and rewards.py
# 2. Test locally
rnow test -n 3 --verbose

# 3. Train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | RL training config |
| `train.jsonl` | Training prompts with reward assignments |
| `rewards.py` | Reward function definitions |
| `requirements.txt` | Python dependencies (e.g., math-verify) |

## train.jsonl Format

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}], "rewards": ["accuracy"], "metadata": {"answer": "4"}}
```

## rewards.py Example

```python
from rnow.core import reward, RewardArgs

@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    response = messages[-1]["content"]
    expected = args.metadata["answer"]
    return 1.0 if expected in response else 0.0
```
