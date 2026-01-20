# ReinforceNow Project

This is a ReinforceNow RL/SFT training project.

## Environment Setup

Dependencies are managed with **uv** (should already be installed with Python 3.11).

```bash
# Activate the virtual environment
source .venv/bin/activate

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

## Quick Commands

```bash
# Test rollouts locally
uv run rnow test -n 3 --verbose

# Submit training run
uv run rnow run

# Check status
uv run rnow status

# Stop a run
uv run rnow stop <RUN_ID>
```

## Skills Available

Claude Code has access to ReinforceNow skills for:
- **rnow-cli** - CLI commands and workflows
- **rnow-config** - Configuration options for config.yml
- **rnow-dataset** - Converting HuggingFace datasets to train.jsonl
- **rnow-rewards** - Writing reward functions
- **rnow-tools** - Writing tool functions
