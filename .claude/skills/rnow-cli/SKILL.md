---
name: rnow-cli
description: Use the ReinforceNow CLI for RLHF training. Use when running rnow commands, initializing projects, submitting training runs, testing rollouts, or downloading models. Triggers on "rnow", "rnow init", "rnow run", "rnow test", "rnow download", "rnow login", "training run".
allowed-tools: Bash, Read, Grep, Glob
---

# ReinforceNow CLI Reference

The `rnow` CLI manages RLHF training projects on the ReinforceNow platform.

## Installation

```bash
pip install rnow
```

## Command Overview

| Command | Description |
|---------|-------------|
| `rnow login` | Authenticate with the platform |
| `rnow logout` | Remove credentials |
| `rnow status` | Check auth and running jobs |
| `rnow orgs` | Manage organizations |
| `rnow init` | Create new project from template |
| `rnow run` | Submit training run |
| `rnow stop` | Cancel active run |
| `rnow test` | Test rollouts locally |
| `rnow download` | Download trained model |

---

## rnow login

Authenticate using OAuth device flow.

```bash
rnow login [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--force` | Force new login even if already authenticated |
| `--api-url URL` | Custom API base URL |

**Example:**
```bash
rnow login
# Opens browser for authentication
# Stores credentials in ~/.reinforcenow/credentials.json
```

---

## rnow logout

Remove stored credentials.

```bash
rnow logout
```

---

## rnow status

Check authentication status and running jobs.

```bash
rnow status
```

**Output:**
```
Logged in as: user@example.com
Organization: My Team (org_abc123)
Active runs: 2
  - run_xyz789 (running) - Math Training
  - run_def456 (queued) - Code Agent
```

---

## rnow orgs

List or select organizations.

```bash
# List all organizations
rnow orgs

# Select an organization
rnow orgs ORG_ID
```

**Example:**
```bash
rnow orgs
# Output:
# * org_abc123 - My Team (owner)
#   org_def456 - Other Team (member)

rnow orgs org_def456
# Switched to: Other Team
```

---

## rnow init

Initialize a new project from a template.

```bash
rnow init [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--template NAME` | Template to use (see below) |
| `--name NAME` | Project name (prompts if not provided) |

### Available Templates

| Template | Type | Description |
|----------|------|-------------|
| `start` | RL | Default single-turn RL (alias for rl-single) |
| `first-rl` | RL | Config only - for first-time users with Claude Code |
| `rl-single` | RL | Single-turn with math reasoning |
| `rl-tools` | RL | Multi-turn with tool calling |
| `sft` | SFT | Supervised finetuning |
| `tutorial-reward` | RL | Learn reward functions |
| `tutorial-tool` | RL | Learn tool functions |
| `mcp-tavily` | RL | External MCP server (web search) |
| `deepseek-aha` | RL | DeepSeek aha-moment training |
| `finqa` | RL | Financial QA |
| `convfinqa` | RL | Conversational financial QA |
| `quantqa` | RL | Quantitative finance |
| `new` | RL | Minimal template |
| `blank` | - | Empty (config only) |

**Examples:**
```bash
# Create SFT project
rnow init --template sft --name "my-sft-project"

# Create RL project with tools
rnow init --template rl-tools

# Create from tutorial
rnow init --template tutorial-reward
```

### Generated Files

| Template | Files |
|----------|-------|
| `sft` | config.yml, train.jsonl |
| `rl-single` | config.yml, train.jsonl, rewards.py, requirements.txt |
| `rl-tools` | config.yml, train.jsonl, rewards.py, tools.py, requirements.txt |
| `blank` | config.yml |

---

## rnow run

Submit project for training.

```bash
rnow run [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--dir PATH` | Project directory (default: current) |
| `--name NAME` | Custom run name |

**Required files:**
- `config.yml` - Configuration
- `train.jsonl` - Training data
- `rewards.py` - Reward functions (RL only)

**Optional files:**
- `tools.py` - Tool definitions
- `requirements.txt` - Python dependencies

**CLI Output:**
```
Run started successfully ✅
  Project: Math Reasoning
  Model: Qwen/Qwen3-8B (thinking: medium)
  Run ID: run_abc123xyz

View your experiment here:
https://www.reinforcenow.ai/runs/run_abc123xyz
```

### How to Respond After Starting a Run

When you run `rnow run` and it succeeds, tell the user:

1. **Confirm success** - The run has started
2. **Share the dashboard link** - Where they can monitor progress
3. **Set expectations** - Training takes time, they can watch metrics live

**Example response:**

> Your training run has started! You can monitor its progress here:
> https://www.reinforcenow.ai/runs/run_abc123xyz
>
> The dashboard will show live metrics, reward curves, and sample outputs as training progresses. Training typically takes 30 minutes to several hours depending on your dataset size and configuration.

### Multi-Model Training

If `model.path` is a list in config.yml:

```yaml
model:
  path:
    - Qwen/Qwen3-4B-Instruct-2507
    - Qwen/Qwen3-8B
```

```bash
rnow run
# Submitting 2 training runs...
# Run 1: run_abc123 (Qwen/Qwen3-4B-Instruct-2507)
# Run 2: run_def456 (Qwen/Qwen3-8B)
```

---

## rnow stop

Cancel an active training run.

```bash
rnow stop RUN_ID
```

**Example:**
```bash
rnow stop run_abc123xyz
# Are you sure you want to stop run_abc123xyz? [y/N]: y
# Run stopped.
# Duration: 2h 15m
# Cost: $12.50
```

---

## rnow test

Test RL rollouts locally before submitting.

```bash
rnow test [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir PATH` | . | Project directory |
| `-n, --num-rollouts N` | 1 | Number of rollouts |
| `--multi-turn/--single-turn` | multi | Allow multi-turn |
| `--with-tools/--no-tools` | with | Enable tools |
| `--model MODEL` | config | Override model |
| `--entry INDICES` | random | Test specific entries (e.g., "0,2,5") |
| `-v, --verbose` | off | Detailed output |
| `--output-dir DIR` | - | Save results as JSON |
| `--truncate N` | - | Truncate output to N chars |
| `--timeout MINS` | 10 | Timeout in minutes |
| `--tinker-api` | off | Use Tinker API |
| `--id ID` | - | Fetch existing rollout |
| `--store` | off | Store rollout ID |

### Examples

**Basic test:**
```bash
rnow test
# Runs 1 rollout, shows reward breakdown
```

**Multiple rollouts with verbose output:**
```bash
rnow test -n 5 --verbose
# Shows full conversation and tool calls for each rollout
```

**Test specific entries:**
```bash
rnow test --entry 0,3,7 --verbose
# Tests entries at indices 0, 3, and 7 from train.jsonl
```

**Override model:**
```bash
rnow test --model gpt-5-nano -n 3
# Uses gpt-5-nano instead of config.model.path
```

**Save results:**
```bash
rnow test -n 10 --output-dir ./results
# Saves each rollout as JSON in ./results/
```

**Single-turn only:**
```bash
rnow test --single-turn
# Forces single turn even if config allows multi-turn
```

**Disable tools:**
```bash
rnow test --no-tools
# Runs without tool calling
```

### Test Output

```
Rollout 1/3
Entry: 0
Prompt: What is 2+2?

Turn 1:
  Assistant: The answer is 4.

Rewards:
  accuracy: 1.0
  format_check: 1.0
Total: 1.0

---
Rollout 2/3
...
```

---

## rnow download

Download a trained model checkpoint.

```bash
rnow download RUN_ID [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output DIR` | ./model | Output directory |

**Example:**
```bash
rnow download run_abc123xyz -o ./my-model
# Downloading checkpoint...
# Progress: 100%
# Saved to: ./my-model/
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REINFORCE_API_URL` | Custom API base URL |
| `RNOW_API_URL` | Alternative API URL variable |

```bash
REINFORCE_API_URL=http://localhost:3000/api rnow test
```

---

## Workflow Examples

### Complete RL Training Workflow

```bash
# 1. Login
rnow login

# 2. Create project
mkdir my-rl-project && cd my-rl-project
rnow init --template rl-single --name "Math Reasoning"

# 3. Edit files as needed
# - config.yml: Adjust batch_size, epochs, etc.
# - rewards.py: Customize reward logic
# - train.jsonl: Add more training data

# 4. Test locally
rnow test -n 5 --verbose

# 5. Submit training
rnow run

# 6. Monitor (check dashboard URL from output)

# 7. Download trained model
rnow download run_abc123xyz -o ./trained-model
```

### Testing Different Configurations

```bash
# Test with different models
rnow test --model gpt-5-nano -n 3
rnow test --model Qwen/Qwen3-8B -n 3

# Compare single vs multi-turn
rnow test --single-turn -n 5
rnow test --multi-turn -n 5

# Test specific problematic entries
rnow test --entry 42,87,103 --verbose
```

### Debugging Failed Runs

```bash
# Check status
rnow status

# Stop failed run
rnow stop run_abc123xyz

# Test locally with verbose output
rnow test -n 1 --verbose --entry 0

# Check specific entry that might be failing
rnow test --entry 42 --verbose
```

---

## Troubleshooting

### "Not authenticated"
```bash
rnow login
```

### "Organization not found"
```bash
rnow orgs  # List available orgs
rnow orgs ORG_ID  # Select the right one
```

### "Validation error"
```bash
# Check your files
rnow test -n 1 --verbose
# Look for specific errors in output
```

### "sandbox=True requires docker field"
Add `"docker": "python:3.11-slim"` to entries using sandbox rewards/tools.

### "batch_size * group_size exceeds limit"
Reduce `batch_size` or `group_size` so product is <= 2048.

### "Model not supported"
Check supported models list or use a checkpoint ID from a previous run.

### Test times out
```bash
rnow test --timeout 30  # Increase to 30 minutes
```

---

## Quick Reference

```bash
# Auth
rnow login
rnow logout
rnow status
rnow orgs [ORG_ID]

# Projects
rnow init --template <name> [--name <project>]
rnow run [--dir <path>] [--name <run-name>]
rnow stop <RUN_ID>

# Testing
rnow test [-n <count>] [--verbose] [--entry <indices>]
rnow test --model <model> --output-dir <dir>

# Models
rnow download <RUN_ID> [-o <output-dir>]
```
