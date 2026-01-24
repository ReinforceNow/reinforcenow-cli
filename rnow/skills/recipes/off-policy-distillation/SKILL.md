---
name: off-policy-distillation
description: Generate training data from teacher models for off-policy distillation. Use when generating teacher completions with tools, preparing data for knowledge distillation, or creating SFT datasets from stronger models. Triggers on "distillation", "teacher model", "generate completions", "off-policy", "knowledge transfer".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Off-Policy Distillation

Generate training data by running a teacher model (e.g., GPT-4o) with tools via `rnow test`, then train a student model via SFT.

> **IMPORTANT**: This is SFT training - **NO `rewards.py` needed**. The student learns by imitating teacher responses.

## Quick Start

```bash
# 1. Run teacher rollouts (saves to rollouts/ folder)
rnow test -n 100 --model gpt-5.2

# 2. Convert rollouts to train.jsonl
python convert_rollouts.py

# 3. Update config for SFT
# config.yml: dataset_type: sft

# 4. Train
rnow run
```

---

## How It Works

1. **Prepare prompts** - Create `train.jsonl` with initial prompts
2. **Run teacher rollouts** - `rnow test` runs the teacher model with your tools/MCP
3. **Convert rollouts** - `convert_rollouts.py` creates SFT-ready `train.jsonl`
4. **Train with SFT** - Student learns to imitate the teacher's responses AND tool use

---

## Step 1: Prepare Prompts

Create a `train.jsonl` with just the initial prompts (no rewards needed):

```json
{"messages": [{"role": "system", "content": "You are a research assistant..."}, {"role": "user", "content": "Find info about NCT01234567"}], "metadata": {"expected_answer": "pembrolizumab"}}
{"messages": [{"role": "system", "content": "You are a research assistant..."}, {"role": "user", "content": "What is the FDA approval date for Keytruda?"}], "metadata": {"expected_answer": "2014-09-04"}}
```

For browser tasks with MCP, include the docker field:
```json
{"messages": [...], "docker": "local/playwright", "metadata": {...}}
```

---

## Step 2: Run Teacher Rollouts

```bash
# Basic - run 10 rollouts with GPT-5.2
rnow test -n 10 --model gpt-5.2

# With specific entries
rnow test -e 0,1,2,3,4 --model gpt-5.2

# All entries (process entire train.jsonl)
rnow test -n 1000 --model gpt-5-pro
```

Rollouts are saved to `rollouts/<timestamp>_<id>.json` with full conversations.

### Supported Models for `rnow test`

| Model | Best For |
|-------|----------|
| `gpt-5-nano` | Fast iteration, testing |
| `gpt-5-mini` | Good balance of speed/quality |
| `gpt-5.2` | Better reasoning (recommended) |
| `gpt-5-pro` | Highest quality teacher |

> **Note**: Only gpt-5 models are supported for `rnow test`. GPU models and other OpenAI models (gpt-4o, etc.) are not supported.

---

## Step 3: Convert Rollouts

```bash
# Convert successful rollouts to train.jsonl
python convert_rollouts.py

# Custom output file
python convert_rollouts.py -o teacher_traces.jsonl

# Include failed rollouts too
python convert_rollouts.py --include-failed
```

### Output Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "Find info about NCT01234567"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "browser_navigate", "arguments": "{\"url\": \"https://clinicaltrials.gov/study/NCT01234567\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "# Study NCT01234567\n\nIntervention: Pembrolizumab..."},
    {"role": "assistant", "content": "The active ingredient is pembrolizumab."}
  ],
  "metadata": {"expected_answer": "pembrolizumab"}
}
```

---

## Step 4: Train with SFT

Update `config.yml`:

```yaml
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

Then run:
```bash
rnow run
```

---

## Example: Browser Agent Distillation

```bash
# 1. Create prompts with MCP browser tools
cat > train.jsonl << 'EOF'
{"messages": [{"role": "system", "content": "Answer by browsing the web."}, {"role": "user", "content": "Who won the 2024 Nobel Prize in Physics?"}], "docker": "local/playwright"}
{"messages": [{"role": "system", "content": "Answer by browsing the web."}, {"role": "user", "content": "What is the current price of Bitcoin?"}], "docker": "local/playwright"}
EOF

# 2. Run teacher with MCP browser
rnow test -n 2 --model gpt-5.2

# 3. Convert and train
python convert_rollouts.py
# Edit config.yml: dataset_type: sft
rnow run
```

---

## Tips

### 1. Use Metadata for Quality Filtering

Include expected answers to filter good examples later:
```json
{"messages": [...], "metadata": {"expected_answer": "42"}}
```

### 2. Batch Processing

Run in batches for large datasets:
```bash
for i in $(seq 0 100 1000); do
  rnow test -e $(seq -s, $i $((i+99))) --model gpt-5.2
done
python convert_rollouts.py
```

### 3. Incremental Generation

Rollouts accumulate in `rollouts/`. Run `rnow test` multiple times, then convert all at once.

### 4. Filter by Quality

After generation, you can filter the output:
```python
import json
entries = [json.loads(l) for l in open("train.jsonl")]
# Keep only entries where assistant gave a final answer
filtered = [e for e in entries if not e["messages"][-1].get("tool_calls")]
```

---

## Troubleshooting

### No rollouts generated
- Check `rnow test` output for errors
- Verify OPENAI_API_KEY is set in `.env`

### Empty train.jsonl
- Run `python convert_rollouts.py --include-failed` to see all rollouts
- Check `rollouts/*.json` files manually

### MCP/Docker issues
- Verify `Dockerfile.playwright` exists in project
- Check `config.yml` has `mcp_url: localhost:8931`

---

## Related Skills

- **rnow-config** - Configure SFT training
- **rnow-train-jsonl** - train.jsonl format details
- **rnow-tools** - Writing custom tools
