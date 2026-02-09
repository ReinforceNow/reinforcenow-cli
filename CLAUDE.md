# ReinforceNow CLI

Python CLI (`rnow`) for RLHF/SFT/Distillation training on the ReinforceNow platform.

## Rules

1. Before using any `rnow` CLI command, check the skill documentation first (if you haven't already).
2. **HuggingFace token warning**: If you see "You are sending unauthenticated requests to the HF Hub", ask the user to add `HF_TOKEN=hf_xxx` to their `.env` file for faster download rates.
3. **ReinforceNow datasets** (e.g., `ReinforceNow/rl-single-math-reasoning`): Skip `rnow test`, go directly to `rnow run`.

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

## Skills

All Claude Code skills live in `rnow/skills/`:

| Skill | Purpose |
|-------|---------|
| `rnow-cli` | CLI commands: init, run, test, stop, download, eval |
| `rnow-config` | Configure training runs (config.yml), convert HF datasets |
| `rnow-rewards` | Write `@reward` functions for RL training |
| `rnow-tools` | Write `@tool` functions for agent training |
| `rnow-train-jsonl` | Format train.jsonl training data |
| `dockerfile` | Write Dockerfiles for sandbox environments |
| `off-policy-distillation` | Generate SFT data from teacher models (OpenRouter) |
| `on-policy-distillation` | Token-level KL distillation with HF teacher models |

## Project Files

- `config.yml` - Training configuration
- `train.jsonl` - Training data (one JSON object per line)
- `rewards.py` - Reward functions (RL only)
- `tools.py` - Tool definitions (optional)
- `requirements.txt` - Python dependencies (optional)

## Directory Structure

```
rnow/
├── cli/commands.py      # CLI entry points (click commands)
├── core/                # @reward, @tool decorators, llm_judge
├── models.py            # Config validation (Pydantic), RewardArgs, ToolArgs
├── skills/              # Claude Code skill documentation
├── templates/           # Project templates for rnow init
└── test.py              # Local test runner
```
