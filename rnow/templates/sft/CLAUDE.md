# SFT (Supervised Fine-Tuning) Template

Train a model on labeled conversation data using next-token prediction.

## Quick Start

```bash
# 1. Prepare train.jsonl with conversations
# 2. Train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | SFT training config |
| `train.jsonl` | Conversation data |

## train.jsonl Format

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

## Key Differences from RL

- No `rewards.py` needed
- No `rewards` field in train.jsonl
- Model learns to imitate assistant responses directly
