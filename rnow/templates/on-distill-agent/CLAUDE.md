# On-Policy Distillation Agent (MedBrowseComp)

Train a medical research agent using **token-level KL distillation** with real web browsing.

## Important: Dataset Processing Default

**When converting datasets, ALWAYS process the ENTIRE train split by default** unless the user explicitly requests a smaller subset. Do NOT suggest using `--limit` or subsets unless the user asks.

For example:
- ✅ `python convert_dataset.py` (processes all 605 samples)
- ❌ `python convert_dataset.py --limit 50` (only if user requests)

This ensures users get the full training benefit from their data.

## Setup

This template requires **Python 3.11+**. Use `uv` to manage dependencies.

```bash
# 1. Create venv with Python 3.11+ using uv
uv venv --python 3.11

# 2. Activate the venv
source .venv/bin/activate

# 3. Install dependencies for dataset conversion
uv pip install datasets
```

## Quick Start

```bash
# 1. Convert the dataset (605 samples)
python convert_dataset.py

# 2. Run training
rnow run
```

## Docker Sandbox for Web Browsing

This template uses **local Dockerfiles** for Docker sandboxes:
- `tools.py` uses `@tool(sandbox=True)` to run in isolated containers
- `train.jsonl` entries have `"docker": "local/crawl4ai"` field
- `Dockerfile.crawl4ai` in project root starts the crawl4ai HTTP server
- Modal builds the image automatically - no need to push to a registry

## Dataset

Uses [MedBrowseComp](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp) - a medical information-seeking benchmark:
- Finding drug ingredients from clinical trials
- Looking up FDA approval information
- Finding patent expiration dates
- Getting stock prices on approval dates

## The `browse` Tool

This template uses **Crawl4AI** (via Docker sandbox) to provide a real web browsing tool:

```python
@tool(sandbox=True)  # Runs in Docker container built from Dockerfile.crawl4ai
def browse(url: str) -> str:
    """Browse a URL and return its content as markdown."""
    # Uses HTTP API: POST http://localhost:11235/crawl
```

The student model learns to call `browse()` with URLs like:
- `https://clinicaltrials.gov/study/NCT01234567`
- `https://www.accessdata.fda.gov/scripts/cder/daf/`

And receives **real page content** as LLM-friendly markdown.

**How it works:**
- `sandbox=True` runs the tool in a per-entry Docker container
- `"docker": "local/crawl4ai"` tells Modal to build from `Dockerfile.crawl4ai`
- The Dockerfile starts crawl4ai's HTTP server before running tools
- Tools call the server via `http://localhost:11235/crawl`

## How It Works (On-Policy Distillation)

```
Training Loop:
1. Student generates response (may include tool calls)
2. Tools execute (browse fetches real web content in Docker sandbox)
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
| `train.jsonl` | Sample data with `docker` field (run convert_dataset.py for full) |
| `convert_dataset.py` | Convert HuggingFace dataset |
| `tools.py` | Crawl4AI `browse` tool with `sandbox=True` |
| `Dockerfile.crawl4ai` | Local Dockerfile that starts crawl4ai server |

Note: `rewards.py` is NOT used - distillation uses teacher KL penalty instead.
Note: `requirements.txt` is NOT needed - dependencies are in the Docker image.

## Config Options

```yaml
dataset_type: distill  # Enables distillation mode

teacher:
  path: Qwen/Qwen3-32B  # Teacher model (use HuggingFace ID)

algorithm:
  kl_penalty_coef: 0.1  # β - KL penalty weight

rollout:
  max_turns: 4  # Allow multiple browse iterations
```

**Note:** Agentic mode is auto-detected based on `max_turns > 1`.

**train.jsonl format:**
```json
{"messages": [...], "tools": ["browse"], "docker": "local/crawl4ai", "metadata": {...}}
```

The `docker` field specifies `local/crawl4ai` which tells Modal to build from `Dockerfile.crawl4ai` in the project directory.

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
