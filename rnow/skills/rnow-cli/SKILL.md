---
name: rnow-cli
description: Use the ReinforceNow CLI for RLHF training. Use when running rnow commands, initializing projects, submitting training runs, testing rollouts, running evals, or downloading models. Triggers on "rnow", "rnow init", "rnow run", "rnow test", "rnow eval", "rnow download", "rnow deploy", "rnow login", "training run".
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
| `rnow eval` | Run model evaluation (pass@k) |
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
| `rl-browser` | RL | Browser agent with Playwright |
| `sft` | SFT | Supervised finetuning |
| `tutorial-reward` | RL | Learn reward functions |
| `tutorial-tool` | RL | Learn tool functions |
| `mcp-tavily` | RL | MCP integration (web search) |
| `kernel` | RL | VLM browser agent with Kernel |
| `skyrl-sql` | RL | SQL reasoning with SkyRL |
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

---

## rnow run

Submit project for training. Supports inline overrides for any config.yml setting.

```bash
rnow run [OPTIONS] [OVERRIDES...]
```

### Options

| Option | Description |
|--------|-------------|
| `-d, --dir PATH` | Project directory (default: current) |
| `-n, --name NAME` | Custom run name |
| `-m, --model MODEL` | Override model path (base model, finetuned UUID, or OpenAI model) |
| `-e, --epochs N` | Override number of epochs |
| `-b, --batch-size N` | Override batch size (1-32) |
| `--lr RATE` | Override learning rate |
| `--debug` | Upload files but don't start training |

### Inline overrides

Any config.yml setting can be overridden with `key=value` arguments. These are applied to the uploaded config (not just in-memory), so the trainer sees the overridden values.

```bash
# Model override (same as -m flag)
rnow run model.path=Qwen/Qwen3-4B-Instruct-2507
rnow run -m Qwen/Qwen3-4B-Instruct-2507  # equivalent shorthand

# Training overrides
rnow run data.batch_size=8 data.group_size=16 trainer.num_epochs=5
rnow run algorithm.adv_estimator=grpo trainer.learning_rate=0.0002

# Rollout overrides (RL only)
rnow run rollout.max_turns=3 rollout.max_context_window=16384
rnow run rollout.reasoning_mode=disabled  # Turn off reasoning for hybrid models

# LoRA rank override
rnow run model.qlora_rank=16
```

### Common overrides

| Override | Description |
|----------|-------------|
| `model.path` | Model to train (e.g., `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-Instruct-2507`) |
| `model.qlora_rank` | LoRA rank (default: 32) |
| `data.batch_size` | Batch size (1-32) |
| `data.group_size` | Rollouts per prompt for RL (1-64) |
| `trainer.num_epochs` | Number of training epochs |
| `trainer.learning_rate` | Learning rate (default: 0.0001) |
| `algorithm.adv_estimator` | Advantage estimator: `grpo`, `gae`, `reinforce` |
| `algorithm.loss_fn` | Loss function: `ppo`, `importance_sampling` |
| `rollout.max_turns` | Max conversation turns for RL |
| `rollout.max_context_window` | Max context window (default: 32768) |
| `rollout.reasoning_mode` | Reasoning mode: `disabled`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |

### Multi-model training

If `model.path` is a list in config.yml, a separate run is submitted for each model:

```yaml
model:
  path:
    - Qwen/Qwen3-8B
    - Qwen/Qwen3-4B-Instruct-2507
    - meta-llama/Llama-3.1-8B-Instruct
```

### Required files
- `config.yml` - Configuration
- `train.jsonl` - Training data
- `rewards.py` - Reward functions (RL only)

### Optional files
- `tools.py` - Tool definitions
- `requirements.txt` - Python dependencies

### Examples

```bash
# Basic run
rnow run

# Override model and epochs
rnow run -m Qwen/Qwen3-4B-Instruct-2507 -e 3

# Train a finetuned model (continue training from checkpoint)
rnow run -m 37e6f995-0efd-4fc6-beaa-badb7af94054

# Override multiple settings inline
rnow run data.batch_size=8 trainer.learning_rate=0.0002 rollout.max_turns=5
```

---

## rnow stop

Cancel an active training run.

```bash
rnow stop RUN_ID [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--save-model` | Save model checkpoint before stopping |
| `-y, --yes` | Skip confirmation |

**Example:**
```bash
rnow stop run_abc123xyz
# Are you sure you want to stop this training run? [y/N]: y
# Save model checkpoint? [y/N]: n
# Run stopped.

rnow stop run_abc123xyz --save-model -y
# Stopping and saving checkpoint...
```

---

## rnow test

Test RL rollouts before submitting. Runs rollouts via the cloud (Modal), not locally.

```bash
rnow test [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir PATH` | . | Project directory |
| `-n, --num-rollouts N` | 1 | Number of rollouts |
| `-e, --entry INDICES` | random | Test specific entries (e.g., `-e 0,2,5` or `-e 0 -e 2 -e 5`) |
| `--all` | false | Test all entries in train.jsonl |
| `--model MODEL` | gpt-5-nano | Override model (OpenAI, GPU, or finetuned UUID) |
| `--max-context-window N` | config | Override max context window |

### Supported models for `--model`

- **OpenAI models**: `gpt-5-nano` (default, fastest), `gpt-5-mini`, `gpt-5.2`, `gpt-5.2-pro`
- **GPU models**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-Instruct-2507`, etc.
- **Finetuned models**: UUID (e.g., `37e6f995-0efd-4fc6-beaa-badb7af94054`)

### Examples

```bash
# Basic test (1 rollout, random entry, gpt-5-nano)
rnow test

# Multiple rollouts
rnow test -n 5

# Test specific entries
rnow test -e 0,3,7

# Test all entries
rnow test --all

# Override model
rnow test --model gpt-5.2 -n 3

# Test finetuned model
rnow test --model 37e6f995-0efd-4fc6-beaa-badb7af94054 -n 3

# Test GPU model
rnow test --model Qwen/Qwen3-8B -n 3
```

---

## rnow eval

Run model evaluation and calculate pass@k metrics.

```bash
rnow eval [OPTIONS]
```

**Two modes:**

### 1. From project directory (uploads files)

```bash
rnow eval --model Qwen/Qwen3-8B --pass1 --pass8
```

Uses `config.yml`, `train.jsonl`, `rewards.py`, `tools.py` from the project directory.

### 2. From existing eval (reuses files)

```bash
rnow eval --eval-id cmle8ma5h000004l44thj3t3a --model gpt-5-nano --pass1
```

Reuses all files from the source eval. Only need to specify `--model`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir PATH` | . | Project directory |
| `-m, --model MODEL` | config | Model to evaluate (name or finetuned UUID) |
| `--eval-id ID` | - | Re-run existing eval with different model |
| `--pass1/--no-pass1` | true | Calculate pass@1 |
| `--pass4/--no-pass4` | false | Calculate pass@4 |
| `--pass8/--no-pass8` | false | Calculate pass@8 |
| `-n, --max-samples N` | all | Limit number of samples |
| `-t, --temperature` | auto | Sampling temperature |
| `--reasoning-mode` | null | `disabled`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (see below) |
| `--max-turns` | config | Max conversation turns |
| `--max-context-window` | config | Max context window tokens |
| `--termination-policy` | config | `last_tool` or `max_turns` |
| `--max-tool-response` | config | Max chars in tool response |
| `--mcp-url` | config | MCP server URL |
| `--max-billing` | null | Max cost in dollars |

### Reasoning Modes

When `--reasoning-mode` is not set (null), each model uses its default:

| Model Family | Default | Supported Efforts |
|-------------|---------|-------------------|
| **GPT-5** (nano, mini) | `medium` (always reasons) | `minimal`, `low`, `medium`, `high` |
| **GPT-5.2** | `none` (no reasoning) | `none`, `low`, `medium`, `high`, `xhigh` |
| **GPT-5.2-pro** | `medium` (always reasons) | `medium`, `high`, `xhigh` |
| **Qwen3 hybrid** (Qwen3-8B, Qwen3-32B) | thinking on | `disabled` to turn off |
| **DeepSeek** (V3.1) | thinking on | `disabled` to turn off |
| **GPT-OSS** (gpt-oss-120b, gpt-oss-20b) | `medium` | `disabled`, `low`, `medium`, `high` |
| **Qwen3 Instruct** / **Llama** | no thinking | N/A (no reasoning support) |

Use `disabled` as the universal "turn off reasoning" value (works for all model types).

### OpenAI Model Compatibility

| Model | Temperature | Default Effort | Supported Efforts |
|-------|-------------|----------------|-------------------|
| `gpt-5-nano` | NEVER | `medium` | minimal, low, medium, high |
| `gpt-5-mini` | NEVER | `medium` | minimal, low, medium, high |
| `gpt-5.2` | Only with `none` effort | `none` | none, low, medium, high, xhigh |
| `gpt-5.2-pro` | NEVER | `medium` | medium, high, xhigh |

### Examples

```bash
# Eval finetuned model on existing eval
rnow eval --eval-id cmle8ma5h000004l44thj3t3a \
  --model 37e6f995-0efd-4fc6-beaa-badb7af94054 --pass1

# Eval with reasoning mode
rnow eval --eval-id cmle8ma5h000004l44thj3t3a \
  --model gpt-5.2 --reasoning-mode high --pass1 --pass8

# Eval from project directory with overrides
rnow eval --model Qwen/Qwen3-8B --pass1 --pass8 \
  --max-turns 5 --temperature 0.8

# Limit to first 50 samples
rnow eval --eval-id cmxyz123 --model gpt-5-nano \
  --pass1 --max-samples 50
```

---

## rnow download

Download a trained model checkpoint.

```bash
rnow download MODEL_ID [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output DIR` | ./<model_name>/ | Output directory |
| `--keep-archive` | false | Keep the tar archive after extraction |

**Example:**
```bash
rnow download abc123 -o ./my-model
# Downloading checkpoint...
# Progress: 100%
# Saved to: ./my-model/

rnow download abc123 --keep-archive
# Keeps both extracted files and .tar archive
```

---
