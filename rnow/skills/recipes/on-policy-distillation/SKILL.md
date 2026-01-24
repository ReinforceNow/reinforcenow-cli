---
name: on-policy-distillation
description: Train models using token-level KL distillation with teacher models. Use when you want true on-policy distillation (Thinking Machines approach), token-level KL penalty, or training with Qwen/Llama as teacher. Triggers on "on-policy", "token-level KL", "KL penalty", "Thinking Machines", "teacher logprobs", "distillation training".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# On-Policy Distillation

## Teacher Model Selection

**IMPORTANT:** On-policy distillation requires computing logprobs for the student's generated tokens.

### Use HuggingFace Model IDs for Teachers

For on-policy distillation, use HuggingFace model IDs (e.g., `Qwen/Qwen3-32B`, `meta-llama/Llama-3.3-70B-Instruct`):

```yaml
teacher:
  path: Qwen/Qwen3-32B  # HuggingFace model ID - supports full logprobs
```

### GPT-4o and Proprietary Models

**GPT-4o, Claude, and other proprietary models do NOT support on-policy distillation** because their APIs don't provide logprobs for arbitrary input tokens.

For these models, use **off-policy distillation** instead:
1. Generate teacher completions with `generate_teacher_data.py`
2. Train student on those completions using SFT
3. See the `off-policy-distillation` skill for details

---

## IMPORTANT: Check CLAUDE.md First

**If working with the `on-distill-agent` template, READ THE `CLAUDE.md` FILE FIRST.**

The template's CLAUDE.md contains critical setup instructions including:
- Python 3.11+ requirement (crawl4ai doesn't work with older Python)
- uv venv setup commands
- crawl4ai installation and browser setup

**Always follow CLAUDE.md setup before running any scripts.**

---

On-policy distillation trains a student model while using a teacher model to provide **token-level KL penalty signals** during training. This is the approach described in [Thinking Machines' blog post](https://thinkingmachines.ai/blog/on-policy-distillation).

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     On-Policy Distillation Loop                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Student generates response for prompt                               │
│                    ↓                                                    │
│  2. Teacher scores EACH TOKEN (computes logprobs for student's tokens)  │
│                                                                         │
│                    ↓                                                    │
│  3. Compute token-level reverse KL: log P_student - log P_teacher       │
│                    ↓                                                    │
│  4. Add KL penalty to advantages: advantage -= β * KL                   │
│                    ↓                                                    │
│  5. Update student with PPO/importance sampling                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Difference from Off-Policy

| Aspect | Off-Policy | On-Policy (this) |
|--------|------------|------------------|
| Teacher role | Generates training data | Scores student outputs |
| When teacher runs | Before training | During training |
| Signal type | Supervised labels | Token-level KL penalty |
| Student learns from | Teacher's responses | Its own responses + KL signal |

## Using the on-distill-agent Template

The `on-distill-agent` template is pre-configured for on-policy distillation on the MedBrowseComp dataset with real web browsing via Crawl4AI.

### Setup (REQUIRED)

```bash
# 1. Create venv with Python 3.11+ using uv
uv venv --python 3.11

# 2. Activate the venv
source .venv/bin/activate

# 3. Install all dependencies
uv pip install datasets crawl4ai nest_asyncio

# 4. Setup crawl4ai browser (required once)
crawl4ai-setup
```

### Run Training

```bash
# 1. Convert dataset
python convert_dataset.py

# 2. Train
rnow run
```

---

## Config.yml for On-Policy Distillation

**IMPORTANT:** On-policy distillation requires `dataset_type: distill` (NOT `rl`).

```yaml
dataset_type: distill  # <-- REQUIRED for on-policy distillation

data:
  train_file: train.jsonl
  batch_size: 4
  group_size: 4

model:
  path: Qwen/Qwen3-8B      # Student model
  qlora_rank: 32

# Teacher model - provides token-level KL supervision
# Use HuggingFace model IDs for full logprobs support
teacher:
  path: Qwen/Qwen3-32B     # Teacher model (larger than student)

algorithm:
  loss_fn: ppo
  adv_estimator: grpo
  kl_penalty_coef: 0.1  # β - Token-level KL penalty weight

rollout:
  max_turns: 10
  max_tokens: 4096
  termination_policy: last_tool
```

**Note:** Agentic mode (multi-turn with tools) is **auto-detected** based on whether `tools.py` exists.

### Key Config Differences from RL

| Field | RL Training | On-Policy Distillation |
|-------|-------------|------------------------|
| `dataset_type` | `rl` | `distill` |
| `teacher:` | Not used | Required (teacher model config) |
| `rewards` in train.jsonl | Required (training signal) | Optional (for tracking only) |
| Agentic mode | Always if tools.py | Auto-detected from tools.py |

---

## Supported Teacher Models

For on-policy distillation, use **HuggingFace model IDs**. These support full logprobs computation:

| Model Family | Recommended Teachers |
|--------------|---------------------|
| **Qwen** | `Qwen/Qwen3-32B`, `Qwen/Qwen3-235B-A22B` |
| **Llama** | `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` |
| **DeepSeek** | `deepseek-ai/DeepSeek-V3` |

### NOT Supported for On-Policy

These models **cannot** be used for on-policy distillation (use off-policy instead):

| Provider | Why Not Supported |
|----------|-------------------|
| **OpenAI** (GPT-4o, etc.) | API doesn't return logprobs for input tokens |
| **Anthropic** (Claude) | No logprobs API at all |
| **Google** (Gemini) | No logprobs API at all |

---

## Quick Start

### 1. Create config.yml

```yaml
dataset_type: distill

data:
  train_file: train.jsonl
  batch_size: 4
  group_size: 4

model:
  path: Qwen/Qwen3-8B      # Student model

teacher:
  path: Qwen/Qwen3-32B     # Teacher model (use HuggingFace ID)

algorithm:
  loss_fn: ppo
  kl_penalty_coef: 0.1     # β - KL penalty weight
```

### 2. Run Training

```bash
rnow run
```

---

## How the KL Penalty Works

### The Math

For each token position `t`:

```
KL_t = log P_student(token_t | context) - log P_teacher(token_t | context)
```

This is the **reverse KL divergence** at each token. The total KL for a sequence is the sum over all tokens.

### In the Loss Function

The advantage for each token is adjusted:

```
advantage_t = original_advantage_t - β * KL_t
```

Where `β` is `kl_penalty_coef`. This penalizes the student for deviating from what the teacher would predict.

### Intuition

- If student assigns **higher probability** than teacher → positive KL → penalty
- If student assigns **lower probability** than teacher → negative KL → bonus
- This pushes the student to match the teacher's token distribution

---

## Architecture

### How Teacher Logprobs Work

The training system creates a teacher client that can compute `log P_teacher(token | context)` for any sequence:

```python
# Teacher computes logprobs for student's generated tokens
teacher_logprobs = await teacher_client.compute_logprobs_async(student_sequence)
```

### Model ID Format

Use HuggingFace model IDs (case-sensitive):

```python
# Correct - HuggingFace format (capital letters)
"Qwen/Qwen3-32B"
"meta-llama/Llama-3.3-70B-Instruct"

# Incorrect - lowercase indicates API provider (not supported for on-policy)
"qwen/qwen3-32b"
"openai/gpt-4o"
```

---

## Configuration Options

### config.yml Fields

| Field | Description |
|-------|-------------|
| `teacher.path` | HuggingFace model ID for teacher (e.g., "Qwen/Qwen3-32B") |
| `algorithm.kl_penalty_coef` | Weight of KL penalty (β), default 0.1 |
| `algorithm.loss_fn` | Loss function: "ppo" or "importance_sampling" |
| `rollout.max_tokens` | Max tokens per response |
| `model.qlora_rank` | LoRA rank for student, default 32 |

---

## Datasets

### train.jsonl Format

Same format as RL training. The `rewards` field is **optional** - add it if you want to track task performance (rewards won't affect training, only the dashboard):

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}], "metadata": {"answer": "4"}}
{"messages": [{"role": "user", "content": "What is 2+2?"}], "rewards": ["accuracy"], "metadata": {"answer": "4"}}
```

### Why Add Rewards for Tracking?

Even though rewards don't affect training (teacher KL is the signal), adding them lets you:
- Monitor task accuracy during training on the dashboard
- See reward breakdown in traces
- Compare distillation vs RL performance

### Agentic Datasets

For agentic tasks with tools, include a `tools.py` file. The system auto-detects agentic mode.

---

## Cost

On-policy distillation uses HuggingFace models served on the training infrastructure. Cost is based on compute time, not per-token API calls.

---

## Monitoring

### Metrics Logged

- `teacher_kl` - Average KL divergence across batch
- `teacher_kl/dataset_0` - Per-dataset KL (if multiple datasets)
- Standard RL metrics (rewards, advantages, loss)

### W&B Integration

```python
config = Config(
    ...
    wandb_project="on-policy-distillation",
    wandb_name="qwen3-8b-gpt4o-teacher",
)
```

---

## Troubleshooting

### "Unsupported teacher model"

This error appears when using a model ID that's not in the supported list. On-policy distillation requires HuggingFace model IDs that support full logprobs computation.

**Fix:** Use a supported HuggingFace model ID (e.g., `Qwen/Qwen3-32B`). See the "Supported Teacher Models" section above.

### Want to use GPT-4o as teacher?

GPT-4o and other proprietary models don't support on-policy distillation because their APIs don't provide logprobs for input tokens. Use **off-policy distillation** instead:

1. Generate teacher completions with `generate_teacher_data.py`
2. Set `dataset_type: sft` in config.yml
3. See the `off-policy-distillation` skill

---

## Related

- **off-policy-distillation** - Generate training data from teachers (simpler, no real-time scoring)
- **rnow-rewards** - Write custom reward functions
- **rnow-config** - Training configuration reference
