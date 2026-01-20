# ReinforceNow Project

This is a ReinforceNow RL/SFT training project.

## Rules

1. Before using any `rnow` CLI command, check the skill documentation first (if you haven't already).
2. **ReinforceNow datasets** (e.g., `ReinforceNow/rl-single-math-reasoning`): Skip `rnow test`, go directly to `rnow run`.
3. **HuggingFace token warning**: If you see "You are sending unauthenticated requests to the HF Hub", just continue without asking for `ReinforceNow/*` datasets.

## First-RL Template

This template only contains `config.yml`. You need to create:
- `train.jsonl` - Fetch from the HuggingFace dataset
- `rewards.py` - Create the reward functions for the dataset
- `requirements.txt` - Add dependencies (e.g., `math-verify==0.5.0` for math datasets)

Use the **rnow-dataset** skill to convert HuggingFace datasets to the correct format.

## Environment Setup

Dependencies are managed with **uv** (should already be installed with Python 3.11).

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run rnow commands
uv run rnow test -n 3 --verbose
uv run rnow run
```

## Project Files

- `config.yml` - Training configuration
- `train.jsonl` - Training data (one JSON object per line)
- `rewards.py` - Reward functions (RL only)
- `tools.py` - Tool definitions (optional)
- `requirements.txt` - Python dependencies (optional)
