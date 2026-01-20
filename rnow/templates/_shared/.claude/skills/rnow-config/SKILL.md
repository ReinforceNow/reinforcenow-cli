---
name: rnow-config
description: Configure ReinforceNow training runs with config.yml and train.jsonl. Use when setting up training configuration, choosing models, configuring RL algorithms, rollout settings, or training data format. Triggers on "config.yml", "train.jsonl", "training config", "batch_size", "group_size", "max_turns", "qlora".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# ReinforceNow Configuration

This guide covers config.yml and train.jsonl setup for RL and SFT training.

## Project Structure

```
my_project/
├── config.yml          # Training configuration (required)
├── train.jsonl         # Training data (required)
├── rewards.py          # Reward functions (required for RL)
├── tools.py            # Tool definitions (optional)
└── requirements.txt    # Python dependencies (optional)
```

## config.yml

### Minimal RL Config

```yaml
project_name: "My RL Project"
dataset_type: rl

data:
  train_file: train.jsonl
  batch_size: 4
  group_size: 8

model:
  path: Qwen/Qwen3-8B

trainer:
  num_epochs: 10
  learning_rate: 0.0001
```

### Minimal SFT Config

```yaml
project_name: "My SFT Project"
dataset_type: sft

data:
  train_file: train.jsonl
  batch_size: 4
  val_split: 0.2

model:
  path: Qwen/Qwen3-8B

trainer:
  num_epochs: 10
  learning_rate: 0.0001
```

### Full RL Config (All Options)

```yaml
# Project identification (auto-filled on first run)
project_id: ""
project_name: "My RL Project"
dataset_id: ""
dataset_type: rl
organization_id: ""
description: "Training description"

# Data configuration
data:
  train_file: train.jsonl       # Path to training data
  batch_size: 16                # 1-32, prompts per batch
  group_size: 4                 # 1-64, rollouts per prompt (RL only)
  # NOTE: batch_size * group_size <= 2048

# Model configuration
model:
  path: Qwen/Qwen3-8B           # Model name or checkpoint ID
  qlora_rank: 32                # LoRA rank (model-specific max)
  qlora_alpha: 64               # LoRA alpha (default: rank * 2)
  name: "custom-model-name"     # Optional output name
  description: "Model desc"     # Optional description

# RL algorithm (RL only)
algorithm:
  loss_fn: ppo                  # 'ppo' or 'importance_sampling'
  adv_estimator: grpo           # 'grpo', 'gae', or 'reinforce'
  kl_penalty_coef: 0.01         # KL divergence penalty

# Rollout configuration (RL only)
rollout:
  max_turns: 1                  # Max conversation turns
  max_tokens: 2048              # Max tokens per generation
  termination_policy: last_tool # 'last_tool' or 'max_turns'
  thinking_mode: null           # null, 'disabled', 'easy', 'medium', 'hard'
  mcp_url: null                 # MCP server URL(s)
  tool_timeout: 60              # Tool execution timeout
  max_tool_response_chars: 4000 # Truncate tool responses (null to disable)
  include_thinking: false       # Include <think> in history

# Training configuration
trainer:
  num_epochs: 30                # Number of epochs
  learning_rate: 0.0001         # Learning rate
  save_step: 20                 # Save checkpoint every N steps (0 = end of epoch)
  eval_step: 0                  # Evaluate every N steps (0 = end of epoch)
```

## Configuration Sections

### data

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `train_file` | Yes | train.jsonl | Path to training data |
| `batch_size` | Yes | - | Prompts per batch (1-32) |
| `group_size` | RL only | 4 | Rollouts per prompt (1-64) |
| `val_split` | SFT only | 0.0 | Validation split ratio (0.0-1.0) |

**Important**: `batch_size * group_size` must be <= 2048 (concurrency limit).

### model

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `path` | Yes | - | Model name or checkpoint ID |
| `qlora_rank` | No | 32 | LoRA rank for efficient finetuning |
| `qlora_alpha` | No | rank * 2 | LoRA alpha scaling |

#### Supported Models

**Qwen (Text)**
- `Qwen/Qwen3-8B` (max rank: 128)
- `Qwen/Qwen3-4B-Instruct-2507` (max rank: 128)
- `Qwen/Qwen3-30B-A3B` (max rank: 64)
- `Qwen/Qwen3-30B-A3B-Instruct-2507` (max rank: 64)
- `Qwen/Qwen3-32B` (max rank: 128)
- `Qwen/Qwen3-235B-A22B-Instruct-2507` (max rank: 64)

**Qwen (Vision)**
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-235B-A22B-Instruct`

**Meta Llama** (max rank: 128)
- `meta-llama/Llama-3.3-70B-Instruct`
- `meta-llama/Llama-3.1-70B`
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B`
- `meta-llama/Llama-3.2-1B`

**DeepSeek** (max rank: 64)
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.1-Base`

**OpenAI** (max rank: 32)
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

**Moonshot**
- `moonshotai/Kimi-K2-Thinking`

#### Multi-Model Training

Train multiple models with the same config:

```yaml
model:
  path:
    - Qwen/Qwen3-4B-Instruct-2507
    - Qwen/Qwen3-8B
    - Qwen/Qwen3-30B-A3B
  qlora_rank: 32
```

The CLI submits separate runs for each model.

### algorithm (RL only)

| Field | Default | Options |
|-------|---------|---------|
| `loss_fn` | ppo | `ppo`, `importance_sampling` |
| `adv_estimator` | grpo | `grpo`, `gae`, `reinforce` |
| `kl_penalty_coef` | 0.01 | KL divergence penalty weight |

**Recommendations**:
- Default `ppo` + `grpo` works well for most tasks
- Lower `kl_penalty_coef` (0.001) for more exploration
- Higher `kl_penalty_coef` (0.1) for stability

### rollout (RL only)

| Field | Default | Description |
|-------|---------|-------------|
| `max_turns` | 1 | Max conversation turns |
| `max_tokens` | 2048 | Max tokens per generation |
| `termination_policy` | last_tool | When to end episode |
| `thinking_mode` | null | Chain-of-thought mode |
| `mcp_url` | null | MCP server URL(s) |
| `tool_timeout` | 60 | Tool execution timeout |
| `max_tool_response_chars` | 4000 | Truncate tool responses |
| `include_thinking` | false | Keep `<think>` in history |

#### Termination Policies

| Policy | Behavior |
|--------|----------|
| `last_tool` | Episode ends when model responds without tool call |
| `max_turns` | Episode always runs for exactly max_turns |

#### Thinking Mode (Reasoning Models)

For models that support chain-of-thought (`<think>` tags):

| Mode | Description |
|------|-------------|
| `null` | Auto-enable for supported models |
| `disabled` | Explicitly disable reasoning |
| `easy` | Light reasoning |
| `medium` | Moderate reasoning |
| `hard` | Deep reasoning (more tokens) |

**Important**: Reasoning models need higher `max_tokens` (8192-16384).

#### MCP Configuration

Connect to external MCP servers for tools:

```yaml
# Single server
rollout:
  mcp_url: "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY"

# Multiple servers
rollout:
  mcp_url:
    - "https://mcp.tavily.com/mcp/?tavilyApiKey=..."
    - "https://mcp.exa.ai/mcp/?apiKey=..."

# In-sandbox MCP (requires docker in train.jsonl)
rollout:
  mcp_url: localhost:8931
```

### trainer

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `num_epochs` | Yes | - | Number of training epochs |
| `learning_rate` | Yes | - | Learning rate |
| `save_step` | No | 0 | Save every N steps (0 = end of epoch) |
| `eval_step` | No | 0 | Evaluate every N steps (0 = end of epoch) |

---

## train.jsonl Format

One JSON object per line. Each entry is a training example.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | Yes | Conversation array |
| `rewards` | RL only | List of reward function names |
| `metadata` | No | Data accessible via `args.metadata` |
| `variables` | No | Template variables via `args.variables` |
| `tools` | No | Filter which tools are available |
| `docker` | If sandbox | Docker image for sandbox execution |
| `docker_env` | No | Environment variables for sandbox |
| `docker_cmd` | No | Custom entrypoint command |

### Message Roles

| Role | Description |
|------|-------------|
| `system` | System instructions (optional, must be first) |
| `user` | User message (at least one required) |
| `assistant` | Assistant response (for multi-turn context) |
| `tool` | Tool call result (for tool use context) |

### Examples

#### Basic RL Entry

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}], "rewards": ["accuracy"], "metadata": {"answer": "4"}}
```

#### SFT Entry

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

#### With System Prompt

```json
{"messages": [{"role": "system", "content": "You are a math tutor"}, {"role": "user", "content": "Explain fractions"}], "rewards": ["quality"]}
```

#### With Tools

```json
{"messages": [{"role": "user", "content": "Search for AI news"}], "rewards": ["relevance"], "tools": ["web_search"]}
```

#### With Sandbox

```json
{
  "messages": [{"role": "user", "content": "Write and run a Python script"}],
  "rewards": ["code_runs", "output_correct"],
  "tools": ["execute_python"],
  "docker": "python:3.11-slim"
}
```

#### With Custom Docker

```json
{
  "messages": [{"role": "user", "content": "Analyze the data"}],
  "rewards": ["accuracy"],
  "docker": "myorg/custom-image:latest",
  "docker_env": {"DEBUG": "true", "DATA_PATH": "/data"},
  "docker_cmd": ["python", "setup.py", "--init"]
}
```

#### Multi-Turn Context

```json
{
  "messages": [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "What's its population?"}
  ],
  "rewards": ["accuracy"],
  "metadata": {"answer": "2.1 million"}
}
```

### Docker Image Requirements

**CRITICAL**: Custom Docker images must be built for `linux/amd64`:

```bash
# Correct
docker build --platform linux/amd64 -t myorg/image:latest .
docker push myorg/image:latest

# Wrong (will fail on x86_64 servers)
docker build -t myorg/image:latest .
```

---

## Common Configurations

### Math Reasoning

```yaml
project_name: "Math Reasoning"
dataset_type: rl

data:
  train_file: train.jsonl
  batch_size: 8
  group_size: 8

model:
  path: Qwen/Qwen3-8B
  qlora_rank: 64

algorithm:
  loss_fn: ppo
  adv_estimator: grpo
  kl_penalty_coef: 0.01

rollout:
  max_turns: 1
  max_tokens: 8192  # High for reasoning
  thinking_mode: medium

trainer:
  num_epochs: 20
  learning_rate: 0.0001
```

### Agent with Tools

```yaml
project_name: "Search Agent"
dataset_type: rl

data:
  train_file: train.jsonl
  batch_size: 4
  group_size: 4

model:
  path: Qwen/Qwen3-8B

rollout:
  max_turns: 5
  max_tokens: 2048
  termination_policy: last_tool
  tool_timeout: 30

trainer:
  num_epochs: 15
  learning_rate: 0.0001
```

### Code Execution

```yaml
project_name: "Code Agent"
dataset_type: rl

data:
  train_file: train.jsonl
  batch_size: 2
  group_size: 4

model:
  path: Qwen/Qwen3-8B

rollout:
  max_turns: 3
  max_tokens: 4096
  tool_timeout: 120  # Longer for code execution

trainer:
  num_epochs: 10
  learning_rate: 0.0001
```

### SFT for Instruction Following

```yaml
project_name: "Instruction Tuning"
dataset_type: sft

data:
  train_file: train.jsonl
  batch_size: 8
  val_split: 0.1

model:
  path: Qwen/Qwen3-8B
  qlora_rank: 32

trainer:
  num_epochs: 3
  learning_rate: 0.00005
  eval_step: 100
```

---

## Validation Rules

1. **batch_size * group_size <= 2048**
2. **qlora_rank <= model's max rank**
3. **Rewards in train.jsonl must exist in rewards.py**
4. **Tools in train.jsonl must exist in tools.py**
5. **sandbox=True requires docker field**
6. **max_tokens must fit in context window**

## Testing Configuration

```bash
# Validate and test locally
rnow test -n 3 --verbose

# Test specific entries
rnow test --entry 0,1,2

# Override model for testing
rnow test --model gpt-5-nano
```
