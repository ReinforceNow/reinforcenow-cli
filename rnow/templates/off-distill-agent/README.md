# MedBrowseComp Agentic Distillation

Train an agent to browse clinicaltrials.gov using off-policy distillation from a teacher model.

**Dataset**: [AIM-Harvard/MedBrowseComp](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp)

## Quick Start

```bash
# 1. Set API key
export OPENROUTER_API_KEY=sk-or-v1-...

# 2. Convert dataset to prompts
python convert_dataset.py -n 50 -o prompts.jsonl

# 3. Generate teacher rollouts
python generate_distillation_data.py prompts.jsonl -m openai/gpt-4o -c 5

# 4. Train with RL
rnow run
```

## Files

| File | Description |
|------|-------------|
| `convert_dataset.py` | Convert HuggingFace dataset to prompts |
| `generate_distillation_data.py` | Generate agentic rollouts from teacher |
| `tools.py` | Browser tools (navigate, click, search, etc.) |
| `rewards.py` | Accuracy reward for MedBrowseComp |
| `config.yml` | RL training config |

## Usage

```bash
# Convert dataset (choose split)
python convert_dataset.py -n 100 --split MedBrowseComp_50

# Generate with different teacher
python generate_distillation_data.py prompts.jsonl -m anthropic/claude-sonnet-4 -c 3

# Test before training
rnow test -n 5 --verbose
```

## Rewards

- `accuracy` - Checks if expected answer is in response (main reward)
- `tool_usage` - Encourages proper tool use
- `task_completion` - Generic task completion check

## Requirements

```bash
pip install httpx tqdm datasets
```
