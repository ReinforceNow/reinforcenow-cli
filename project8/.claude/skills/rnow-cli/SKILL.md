---
name: rnow-cli
description: Use the ReinforceNow CLI for RLHF training. Use when running rnow commands, initializing projects, submitting training runs, testing rollouts, or downloading models.
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
| `first-rl` | RL | Starter template for RL |
| `rl-single` | RL | Single-turn RL with rewards |
| `rl-tools` | RL | Multi-turn RL with tool calling |
| `sft` | SFT | Supervised finetuning |
| `tutorial-reward` | RL | Learn reward functions |
| `tutorial-tool` | RL | Learn tool functions |
| `mcp-tavily` | RL | MCP integration (web search) |
| `kernel` | RL | VLM browser agent with Kernel |
| `rl-browser` | RL | Browser agent with Playwright |
| `off-distill-agent` | SFT | Off-policy distillation |
| `on-distill-agent` | Distill | On-policy KL distillation |
| `posttrain` | Midtrain | Continued pretraining |

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

**Example:**
```bash
cd my-project
rnow run

# Output:
# Validating project...
# Uploading files...
# Starting run: run_abc123xyz
# View at: https://www.reinforcenow.ai/runs/run_abc123xyz
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
| `--entry INDICES` | random | Test specific entries (e.g., "0,2,5") |
| `--model MODEL` | gpt-5-nano | Override model for testing |

### Available Models

**OpenAI API models (default, fast):**
- `gpt-5-nano` - Fastest, recommended for quick testing
- `gpt-5-mini` - Faster
- `gpt-5.2` - Balanced
- `gpt-5-pro` - Highest quality

**GPU models (slower, uses actual training infrastructure):**

Text models:
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `deepseek-ai/DeepSeek-V3.1`

**VLM models (for vision tasks with screenshots):**
- `Qwen/Qwen3-VL-30B-A3B-Instruct` - Vision-language model
- `Qwen/Qwen3-VL-235B-A22B-Instruct` - Larger VLM

> **Important:** Default `gpt-5-nano` is text-only and cannot process images. For VLM projects that return screenshots (e.g., browser agents), use `--model Qwen/Qwen3-VL-30B-A3B-Instruct` to test with actual vision capabilities.

### Examples

**Basic test:**
```bash
rnow test
# Runs 1 rollout with gpt-5-nano (text-only)
```

**Multiple rollouts:**
```bash
rnow test -n 5
```

**Test specific entries:**
```bash
rnow test --entry 0,3,7
# Tests entries at indices 0, 3, and 7 from train.jsonl
```

**Test with VLM model (for vision/screenshot projects):**
```bash
rnow test --model Qwen/Qwen3-VL-30B-A3B-Instruct -n 1
# Uses actual VLM that can see images
```

**Test with GPU model:**
```bash
rnow test --model Qwen/Qwen3-8B -n 3
# Uses GPU infrastructure instead of OpenAI API
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
