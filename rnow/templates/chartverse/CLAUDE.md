# ChartVerse-RL (Chart Reasoning)

Train a vision-language model on chart understanding with RL.

## Important: Dataset Processing Default

**ALWAYS process the ENTIRE train split (40K samples)** unless the user explicitly requests a smaller subset.

## Quick Start

```bash
# 1. Convert dataset
uv run --with datasets --with pillow python convert_dataset.py

# 2. Run training
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | RL config with Qwen3-VL-30B |
| `convert_dataset.py` | Downloads ChartVerse-RL-40K |
| `rewards.py` | math_verify accuracy reward |
