---
name: rnow-config
description: Configure ReinforceNow training runs with config.yml and train.jsonl. Also covers converting HuggingFace datasets to ReinforceNow format. Triggers on "config.yml", "train.jsonl", "training config", "batch_size", "group_size", "max_turns", "qlora", "HuggingFace", "dataset", "convert dataset".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# ReinforceNow Configuration

This guide covers config.yml and train.jsonl setup for RL, SFT, and Distillation training.

## Project Structure

```
my_project/
├── config.yml          # Training configuration (required)
├── train.jsonl         # Training data (required)
├── rewards.py          # Reward functions (required for RL, not needed for SFT/Distillation)
├── tools.py            # Tool definitions (optional, RL only)
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

### Minimal Distillation Config

On-policy distillation trains a student model to match a teacher model's behavior. The student generates, the teacher grades each token, and KL divergence provides supervision.

```yaml
project_name: "My Distillation Project"
dataset_type: distill

data:
  train_file: train.jsonl
  batch_size: 8
  group_size: 4

model:
  path: Qwen/Qwen3-8B        # Student model

teacher:
  path: Qwen/Qwen3-32B       # Teacher model (larger)

rollout:
  max_tokens: 8192

trainer:
  num_epochs: 3
  learning_rate: 0.0001
```

**Key points:**
- No `rewards.py` needed - teacher provides all supervision via KL penalty
- Student generates on its own distribution (on-policy)
- Teacher computes log probabilities for each token
- KL penalty coefficient is 1.0 (full weight on teacher supervision)

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
  max_context_window: 32768    # Max context window in tokens (tool results auto-truncated)
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

### teacher (Distillation only)

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Teacher model name (must be a supported model) |

The teacher provides supervision via reverse KL divergence. Use a larger/more capable model as teacher:

```yaml
teacher:
  path: Qwen/Qwen3-32B  # 32B teacher distilling to 8B student
```

**Teacher selection tips:**
- Use a model from the same family (e.g., Qwen teacher for Qwen student)
- Larger teachers generally produce better students
- Teacher must be a supported model (see model list above)

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

### rollout (RL and Distillation)

| Field | Default | Description |
|-------|---------|-------------|
| `max_turns` | 1 | Max conversation turns |
| `max_tokens` | 2048 | Max tokens per generation |
| `termination_policy` | last_tool | When to end episode |
| `thinking_mode` | null | Chain-of-thought mode |
| `mcp_url` | null | MCP server URL(s) |
| `tool_timeout` | 60 | Tool execution timeout |
| `max_context_window` | 32768 | Max context window in tokens |
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

For full train.jsonl documentation including message format, sandbox/docker configuration, and examples, see the **rnow-train-jsonl** skill.

---

## Converting HuggingFace Datasets

This section shows how to convert HuggingFace datasets to train.jsonl format.

### SFT Conversion

For SFT, include both user and assistant messages:

```python
from datasets import load_dataset
import json

dataset = load_dataset("your-dataset-name", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        entry = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
        f.write(json.dumps(entry) + "\n")
```

#### Alpaca-style Dataset

```python
from datasets import load_dataset
import json

dataset = load_dataset("tatsu-lab/alpaca", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        if row.get("input"):
            user_content = f"{row['instruction']}\n\nInput: {row['input']}"
        else:
            user_content = row["instruction"]

        entry = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["output"]}
            ]
        }
        f.write(json.dumps(entry) + "\n")
```

#### Multi-turn Conversations

```python
# Input: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
messages = []
for turn in row["conversations"]:
    role = "user" if turn["from"] == "human" else "assistant"
    messages.append({"role": role, "content": turn["value"]})

entry = {"messages": messages}
```

### RL Conversion

For RL, include only the prompt (user message). The model generates responses during training.

```python
from datasets import load_dataset
import json

dataset = load_dataset("your-math-dataset", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        entry = {
            "messages": [
                {"role": "user", "content": row["question"]}
            ],
            "rewards": ["accuracy"],
            "metadata": {
                "expected_answer": row["answer"]
            }
        }
        f.write(json.dumps(entry) + "\n")
```

#### GSM8K Math Dataset

```python
from datasets import load_dataset
import json
import re

dataset = load_dataset("gsm8k", "main", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        # Extract final answer (#### followed by number)
        answer_match = re.search(r"####\s*(.+)$", row["answer"])
        final_answer = answer_match.group(1).strip() if answer_match else row["answer"]

        entry = {
            "messages": [{"role": "user", "content": row["question"]}],
            "rewards": ["accuracy"],
            "metadata": {"expected_answer": final_answer}
        }
        f.write(json.dumps(entry) + "\n")
```

#### MATH Dataset (Competition Math)

```python
from datasets import load_dataset
import json
import re

dataset = load_dataset("hendrycks/competition_math", split="train")

def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1) if match else text

with open("train.jsonl", "w") as f:
    for row in dataset:
        answer = extract_boxed(row["solution"])
        # Wrap in $$ for math-verify (skip if already delimited or plain number)
        if not answer.startswith(("$", "\\(")) and not answer.replace(".", "").replace("-", "").isdigit():
            answer = f"$${answer}$$"

        entry = {
            "messages": [{"role": "user", "content": row["problem"]}],
            "rewards": ["accuracy"],
            "metadata": {"expected_answer": answer}
        }
        f.write(json.dumps(entry) + "\n")
```

**Note**: For math-verify, `expected_answer` MUST have math delimiters (`$...$` or `\(...\)`). Raw LaTeX like `\sqrt{2}` won't parse - use `$\sqrt{2}$`. Plain numbers like `42` work as-is.

For reward function examples (math-verify, llm_judge), see the **rnow-rewards** skill.

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

### Distillation for Reasoning

```yaml
project_name: "Distilled Reasoning Model"
dataset_type: distill

data:
  train_file: train.jsonl
  batch_size: 8
  group_size: 4

model:
  path: Qwen/Qwen3-8B          # Student
  qlora_rank: 32

teacher:
  path: Qwen/Qwen3-32B         # Teacher

rollout:
  max_tokens: 8192             # Enough for reasoning

trainer:
  num_epochs: 3
  learning_rate: 0.0001
  save_step: 20
```

**When to use distillation:**
- Transfer reasoning capabilities from a large model to a smaller one
- Create a cost-effective model that approximates a larger model's behavior
- On-policy distillation avoids exposure bias (student learns from its own mistakes)

---

## Validation Rules

1. **batch_size * group_size <= 2048**
2. **qlora_rank <= model's max rank**
3. **Rewards in train.jsonl must exist in rewards.py** (RL only)
4. **Tools in train.jsonl must exist in tools.py** (RL only)
5. **sandbox=True requires docker field** (RL only)
6. **max_tokens must fit in context window**
7. **Distillation requires teacher section** with valid model path

## Testing Configuration

```bash
# Validate and test locally
rnow test -n 3 --verbose

# Test specific entries
rnow test --entry 0,1,2

# Override model for testing
rnow test --model gpt-5-nano
```
