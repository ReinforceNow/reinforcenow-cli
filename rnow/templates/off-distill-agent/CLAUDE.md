# Off-Policy Distillation

Train a model on teacher-generated completions via SFT.

## Quick Start (HuggingFace Dataset)

```bash
# 1. Convert HuggingFace dataset to train.jsonl
python convert_hf_dataset.py --max-samples 1000

# 2. Train (SFT)
rnow run
```

## Quick Start (Generate with rnow test)

```bash
# 1. Create prompts in train.jsonl (no assistant messages)
# 2. Run teacher rollouts
rnow test -n 100 --model gpt-5.2

# 3. Convert rollouts to train.jsonl
python convert_rollouts.py

# 4. Train (SFT)
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `convert_hf_dataset.py` | Convert HuggingFace dataset to train.jsonl |
| `convert_rollouts.py` | Convert rnow test rollouts to train.jsonl |
| `config.yml` | SFT training config |
| `train.jsonl` | Training data |

## Using HuggingFace Datasets

The default uses [zenml/cuad-deepseek](https://huggingface.co/datasets/zenml/cuad-deepseek) - legal contract classification with DeepSeek-R1 reasoning traces.

```bash
# Different sample sizes
python convert_hf_dataset.py --max-samples 100    # Quick test
python convert_hf_dataset.py --max-samples 5000   # Medium
python convert_hf_dataset.py --max-samples 26000  # Full dataset
```

## Using rnow test

For custom prompts without existing teacher completions:

```bash
# Create train.jsonl with prompts only
{"messages": [{"role": "user", "content": "..."}], "metadata": {...}}

# Generate teacher completions
rnow test -n 50 --model gpt-5.2
rnow test -n 50 --model gpt-5-pro  # Higher quality

# Convert and train
python convert_rollouts.py
rnow run
```

Supported models for `rnow test`: gpt-5-nano, gpt-5-mini, gpt-5.2, gpt-5-pro
