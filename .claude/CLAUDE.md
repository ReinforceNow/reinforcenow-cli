# ReinforceNow Project

This is a ReinforceNow RL/SFT training project.

## Rules

1. Before using any `rnow` CLI command, check the skill documentation first (if you haven't already).
2. **ReinforceNow datasets** (e.g., `ReinforceNow/rl-single-math-reasoning`): Skip `rnow test`, go directly to `rnow run`.

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
