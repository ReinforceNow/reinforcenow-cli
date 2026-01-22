---
name: off-policy-distillation
description: Generate training data from teacher models for off-policy distillation. Use when converting prompts to SFT datasets using larger models, generating teacher completions, or preparing data for knowledge distillation. Triggers on "distillation", "teacher model", "generate completions", "OpenRouter", "off-policy", "knowledge transfer".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Off-Policy Distillation

Off-policy distillation trains a smaller "student" model to imitate a larger "teacher" model by learning from teacher-generated examples.

## How It Works

1. **Start with prompts** - A dataset of user messages/tasks
2. **Generate teacher responses** - Run prompts through a capable teacher model
3. **Train with SFT** - Fine-tune the student on teacher responses

This transfers the teacher's capabilities to a smaller, faster model.

## When to Use

- **Knowledge transfer**: Distill a 70B model's capabilities into an 8B model
- **Cost reduction**: Create a cheaper model that mimics expensive API models
- **Faster inference**: Smaller models for production deployment
- **Domain specialization**: Focus a general model on specific tasks

## Step-by-Step Guide

### Step 1: Prepare Your Prompts

Create a file with your prompts (one per line or JSONL):

**prompts.txt** (simple):
```
What is photosynthesis?
Explain quantum computing in simple terms.
Write a Python function to sort a list.
```

**prompts.jsonl** (with metadata):
```json
{"prompt": "What is photosynthesis?", "category": "science"}
{"prompt": "Explain quantum computing", "category": "tech"}
{"prompt": "Write a sorting function", "category": "code"}
```

### Step 2: Get an OpenRouter API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Create an account and add credits
3. Generate an API key from the dashboard

### Step 3: Set Up Environment

```bash
# Create .env file with your API key
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env

# Install dependencies (if not using rnow's built-in)
pip install httpx python-dotenv tqdm
```

### Step 4: Generate Teacher Completions

Use the provided script to generate completions concurrently:

```bash
# Basic usage
python generate_teacher_data.py \
  --input prompts.txt \
  --output train.jsonl \
  --model anthropic/claude-sonnet-4

# With options
python generate_teacher_data.py \
  --input prompts.jsonl \
  --output train.jsonl \
  --model openai/gpt-5.2 \
  --system "You are a helpful math tutor." \
  --max-tokens 2048 \
  --temperature 0.7 \
  --concurrency 10
```

### Step 5: Configure SFT Training

Create `config.yml`:

```yaml
project_name: "Distilled Model"
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
```

### Step 6: Run Training

```bash
rnow run
```

---

## Script Reference: generate_teacher_data.py

The script handles concurrent API calls to generate teacher completions efficiently.

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input file (txt or jsonl) |
| `--output` | train.jsonl | Output file path |
| `--model` | anthropic/claude-sonnet-4 | Teacher model (OpenRouter format) |
| `--system` | None | System prompt for all completions |
| `--max-tokens` | 2048 | Max tokens per completion |
| `--temperature` | 0.7 | Sampling temperature |
| `--concurrency` | 5 | Parallel requests |
| `--timeout` | 120 | Request timeout in seconds |

### Input Formats

**Plain text** (prompts.txt):
```
First prompt here
Second prompt here
```

**JSONL** (prompts.jsonl):
```json
{"prompt": "First prompt", "system": "Custom system prompt"}
{"prompt": "Second prompt", "metadata": {"category": "math"}}
```

JSONL entries can include:
- `prompt` (required): The user message
- `system` (optional): Per-entry system prompt (overrides --system)
- `metadata` (optional): Preserved in output for use in rewards

### Output Format

Generated `train.jsonl` is ready for SFT:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}], "metadata": {"teacher": "openai/gpt-5.2", "category": "math"}}
```

---

## Recommended Teacher Models

| Model | Best For | Cost |
|-------|----------|------|
| `anthropic/claude-sonnet-4` | General, reasoning | Medium |
| `openai/gpt-5.2` | Broad capabilities | Medium |
| `openai/o3` | Complex reasoning | High |
| `deepseek/deepseek-r1` | Math, code | Low |
| `google/gemini-2.5-pro` | Long context | Medium |

See [openrouter.ai/models](https://openrouter.ai/models) for full list.

---

## Tips for Quality Distillation

### 1. Use High Temperature for Diversity

```bash
python generate_teacher_data.py --temperature 0.9 --input prompts.txt
```

Higher temperature creates more diverse training examples.

### 2. Generate Multiple Completions Per Prompt

Run the script multiple times with different seeds or use `--samples`:

```bash
# Generate 3 completions per prompt
python generate_teacher_data.py --samples 3 --input prompts.txt
```

### 3. Filter Low-Quality Responses

After generation, filter responses that are too short or contain errors:

```python
import json

with open("train.jsonl") as f:
    entries = [json.loads(line) for line in f]

# Filter: keep responses > 100 chars
filtered = [e for e in entries if len(e["messages"][-1]["content"]) > 100]

with open("train_filtered.jsonl", "w") as f:
    for entry in filtered:
        f.write(json.dumps(entry) + "\n")
```

### 4. Include System Prompts

System prompts guide the teacher's behavior:

```bash
python generate_teacher_data.py \
  --system "You are an expert Python programmer. Write clean, well-documented code." \
  --input coding_prompts.txt
```

### 5. Preserve Metadata for RL Fine-Tuning

If you plan to do RL after SFT, include ground truth in metadata:

```json
{"prompt": "What is 2+2?", "metadata": {"expected_answer": "4"}}
```

This metadata flows through to the output for use in reward functions later.

---

## Cost Estimation

Rough estimates for 10,000 prompts (2K tokens output each):

| Model | Input Cost | Output Cost | Total |
|-------|------------|-------------|-------|
| claude-sonnet-4 | ~$3 | ~$15 | ~$18 |
| gpt-5.2 | ~$2.5 | ~$10 | ~$12.5 |
| deepseek-r1 | ~$0.5 | ~$2 | ~$2.5 |

Check [openrouter.ai/models](https://openrouter.ai/models) for current pricing.

---

## Troubleshooting

### Rate Limiting

If you hit rate limits, reduce concurrency:

```bash
python generate_teacher_data.py --concurrency 2 --input prompts.txt
```

### Timeouts

For long completions, increase timeout:

```bash
python generate_teacher_data.py --timeout 300 --max-tokens 4096
```

### API Key Issues

Ensure your `.env` file is in the current directory:

```bash
# Check .env exists and has the key
cat .env | grep OPENROUTER
```

### Resume Interrupted Generation

The script saves progress. To resume, re-run with the same output file - it skips already-generated prompts.

---

## Related Skills

- **rnow-config** - Configure SFT training
- **rnow-train-jsonl** - train.jsonl format details
