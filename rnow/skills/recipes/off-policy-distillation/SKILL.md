---
name: off-policy-distillation
description: Generate training data from teacher models for off-policy distillation. Use when converting prompts to SFT datasets using larger models, generating teacher completions with tools, or preparing data for knowledge distillation. Triggers on "distillation", "teacher model", "generate completions", "OpenRouter", "off-policy", "knowledge transfer".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Off-Policy Distillation

Generate training data by running a teacher model (e.g., GPT-4o) with tools, then train a student model via SFT.

> **IMPORTANT**: This is SFT training - **NO `rewards.py` needed**. The student learns by imitating teacher responses.

## Setup

```bash
# 1. Create venv with Python 3.11+ using uv
uv venv --python 3.11

# 2. Activate the venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install httpx tqdm crawl4ai nest_asyncio

# 4. Setup crawl4ai browser (required once)
crawl4ai-setup

# 5. Set OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
```

## Quick Start

```bash
# 1. Generate teacher completions with tool use
python generate_distillation_data.py prompts.jsonl \
  --model openai/gpt-4o-mini \
  --tools tools.py \
  --output train.jsonl

# 2. Train (SFT - no rewards needed)
rnow run
```

---

## How It Works

1. **Prepare prompts** - A list of tasks/questions
2. **Run agentic rollouts** - Teacher model uses tools (e.g., browse) to solve tasks
3. **Save conversations** - Full multi-turn conversations with tool calls saved to train.jsonl
4. **Train with SFT** - Student learns to imitate the teacher's responses AND tool use

---

## Script: generate_distillation_data.py

Runs concurrent agentic rollouts with adaptive rate limiting.

### Usage

```bash
# Basic - agentic with tools
python generate_distillation_data.py prompts.jsonl --tools tools.py

# No tools - simple completions
python generate_distillation_data.py prompts.jsonl --tools none

# Custom model and concurrency
python generate_distillation_data.py prompts.jsonl \
  --model openai/gpt-4o \
  --concurrency 10 \
  --max-turns 15
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | required | Input prompts file (txt or jsonl) |
| `-o, --output` | train.jsonl | Output file |
| `-t, --tools` | tools.py | Tools file (use 'none' to disable) |
| `-m, --model` | openai/gpt-4o-mini | Teacher model (OpenRouter format) |
| `-s, --system` | None | System prompt |
| `-n, --num` | None | Max prompts to process |
| `-c, --concurrency` | 20 | Initial concurrent requests (adapts on rate limits) |
| `--max-tokens` | 2048 | Max tokens per response |
| `--max-turns` | 10 | Max turns per rollout |
| `--temperature` | 0.7 | Sampling temperature |

### Input Format

**prompts.txt** (simple):
```
Find the active ingredient in NCT01234567
What was the FDA approval date for Keytruda?
```

**prompts.jsonl** (with metadata):
```json
{"prompt": "Find the active ingredient in NCT01234567", "metadata": {"expected_answer": "pembrolizumab"}}
{"prompt": "What was the FDA approval date for Keytruda?", "metadata": {"expected_answer": "2014-09-04"}}
```

### Output Format

Full conversations with tool calls:
```json
{
  "messages": [
    {"role": "user", "content": "Find the active ingredient in NCT01234567"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "browse", "arguments": "{\"url\": \"https://clinicaltrials.gov/study/NCT01234567\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "# Study NCT01234567\n\nIntervention: Pembrolizumab 200mg..."},
    {"role": "assistant", "content": "The active ingredient is pembrolizumab."}
  ],
  "metadata": {"teacher": "openai/gpt-4o-mini", "turns": 2},
  "tools": ["browse"]
}
```

---

## tools.py - Browse Tool

The included `tools.py` uses Crawl4AI for web browsing:

```python
@tool
def browse(url: str) -> str:
    """Browse a URL and return its content as markdown."""
    # Uses Crawl4AI to fetch and convert page to markdown
```

You can add more tools as needed:

```python
@tool
def search(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
```

---

## config.yml for SFT

```yaml
project_name: "Distilled Agent"
dataset_type: sft  # No rewards needed

data:
  train_file: train.jsonl
  batch_size: 4

model:
  path: Qwen/Qwen3-8B
  qlora_rank: 32

trainer:
  num_epochs: 3
  learning_rate: 0.00005
```

**Note**: No `rewards.py` is needed - SFT learns from teacher examples directly.

---

## Recommended Teacher Models

| Model | Best For | Cost |
|-------|----------|------|
| `openai/gpt-4o-mini` | Fast, cheap, good for browsing | Low |
| `openai/gpt-4o` | Better reasoning | Medium |
| `anthropic/claude-sonnet-4` | Complex tasks | Medium |
| `openai/o3` | Deep reasoning | High |

See [openrouter.ai/models](https://openrouter.ai/models) for full list.

---

## Tips

### 1. Use Metadata for Later RL

Include expected answers in metadata for potential RL fine-tuning later:

```json
{"prompt": "What is 2+2?", "metadata": {"expected_answer": "4"}}
```

### 2. Filter Bad Samples

After generation, filter out low-quality samples:

```python
import json
entries = [json.loads(l) for l in open("train.jsonl")]
# Keep only successful rollouts (assistant gave final answer)
filtered = [e for e in entries if not e["messages"][-1].get("tool_calls")]
```

### 3. Resume Interrupted Generation

The script auto-resumes. Just re-run with the same output file.

### 4. Handle Rate Limits

The script automatically reduces concurrency on rate limits. Start high (20) and let it adapt.

---

## Troubleshooting

### Crawl4AI Setup Issues

```bash
# Make sure playwright browsers are installed
playwright install chromium
```

### Rate Limiting

The script handles this automatically. If you see many rate limits, reduce initial concurrency:

```bash
python generate_distillation_data.py prompts.jsonl -c 5
```

### Tool Execution Errors

Check that tools.py dependencies are installed:

```bash
uv pip install crawl4ai nest_asyncio
crawl4ai-setup
```

---

## Related Skills

- **rnow-config** - Configure SFT training
- **rnow-train-jsonl** - train.jsonl format details
- **rnow-tools** - Writing custom tools
