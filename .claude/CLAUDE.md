# ReinforceNow Project

This is a ReinforceNow RL/SFT training project.

## Rules

1. Before using any `rnow` CLI command, check the skill documentation first (if you haven't already).
2. **HuggingFace token warning**: If you see "You are sending unauthenticated requests to the HF Hub", ask the user to add `HF_TOKEN=hf_xxx` to their `.env` file for faster download rates.

## Environment Setup

Dependencies are managed with **uv** (should already be installed with Python 3.11).

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run rnow commands (use --active to ensure uv uses the local venv)
uv run --active rnow test -n 3 --verbose
uv run --active rnow run
```

**Important:** Always use `uv run --active` to ensure uv uses the activated local `.venv` instead of a global environment. Without `--active`, uv may use an older rnow version from a different location.

## Project Files

- `config.yml` - Training configuration
- `train.jsonl` - Training data (one JSON object per line)
- `rewards.py` - Reward functions (RL only)
- `tools.py` - Tool definitions (optional)
- `requirements.txt` - Python dependencies (optional)
