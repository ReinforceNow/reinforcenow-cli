# On-Policy Distillation Agent (MedBrowseComp)

Train a medical research agent using **token-level KL distillation** with real web browsing.

## IMPORTANT: Setup with uv

This template requires **Python 3.11+** and **crawl4ai** for web browsing. Use `uv` to manage dependencies.

**Before running any scripts, execute these setup commands:**

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

## Quick Start

After setup is complete:

```bash
# 1. Convert the dataset (605 samples)
python convert_dataset.py

# 2. Run training
rnow run
```

## Dataset

Uses [MedBrowseComp](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp) - a medical information-seeking benchmark:
- Finding drug ingredients from clinical trials
- Looking up FDA approval information
- Finding patent expiration dates
- Getting stock prices on approval dates

## The `browse` Tool

This template uses **Crawl4AI** to provide a real web browsing tool:

```python
@tool
def browse(url: str) -> str:
    """Browse a URL and return its content as markdown."""
```

The student model learns to call `browse()` with URLs like:
- `https://clinicaltrials.gov/study/NCT01234567`
- `https://www.accessdata.fda.gov/scripts/cder/daf/`

And receives **real page content** as LLM-friendly markdown.

## How It Works (On-Policy Distillation)

```
Training Loop:
1. Student generates response (may include tool calls)
2. Tools execute (browse fetches real web content)
3. Teacher (Qwen3-32B) provides TOKEN-LEVEL supervision via logprobs
4. KL penalty: advantage_t -= β * (log P_student - log P_teacher)
5. Student updated with PPO + KL penalty

Key: ALL supervision comes from teacher KL penalty, NOT reward functions.
```

This is different from RL training which uses reward functions. Distillation learns
to match the teacher's token distribution directly.

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Training config with `dataset_type: distill` |
| `train.jsonl` | Sample data (run convert_dataset.py for full) |
| `convert_dataset.py` | Convert HuggingFace dataset |
| `tools.py` | Crawl4AI `browse` tool |

Note: `rewards.py` is NOT used - distillation uses teacher KL penalty instead.

## Config Options

```yaml
dataset_type: distill  # Enables distillation mode

teacher:
  path: Qwen/Qwen3-32B  # Teacher model (use HuggingFace ID)

algorithm:
  kl_penalty_coef: 0.1  # β - KL penalty weight

rollout:
  max_turns: 10  # Allow multiple browse iterations
```

**Note:** Agentic mode is auto-detected based on `tools.py` existence.

For GPT-4o or other proprietary models, use **off-policy distillation** instead.

## How KL Distillation Works

Unlike RL which uses sparse rewards at the end of episodes, distillation provides
**dense token-level supervision**:

1. Student generates tokens: `t1, t2, t3, ...`
2. Teacher computes logprobs for each token: `log P_teacher(t_i | context)`
3. Student has its own logprobs: `log P_student(t_i | context)`
4. KL penalty per token: `β * (log P_student - log P_teacher)`
5. This penalty is subtracted from advantages during PPO update

The result: student learns to generate tokens that the teacher would also generate,
while still being able to explore and use tools.
