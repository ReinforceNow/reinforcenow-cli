# Off-Policy Distillation Agent (MedBrowseComp)

Train a model on teacher-generated examples using the MedBrowseComp medical QA dataset.

## IMPORTANT: Setup with uv

This template requires **Python 3.11+** and **crawl4ai** for web browsing. Use `uv` to manage dependencies.

**Before running any scripts, execute these setup commands:**

```bash
# 1. Create venv with Python 3.11+ using uv
uv venv --python 3.11

# 2. Activate the venv
source .venv/bin/activate

# 3. Install all dependencies
uv pip install datasets httpx tqdm crawl4ai nest_asyncio

# 4. Setup crawl4ai browser (required once)
crawl4ai-setup
```

## Quick Start

After setup is complete:

```bash
# 1. Set API key (or add to .env file)
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# 2. Download full MedBrowseComp dataset (605 samples)
python convert_dataset.py

# 3. Generate teacher completions with real web browsing
python generate_distillation_data.py prompts.jsonl \
  --model openai/gpt-4o-mini \
  --tools tools.py \
  --output train.jsonl \
  --concurrency 5

# 4. Run training
rnow run
```

## Dataset Info

MedBrowseComp has 605 training samples across 5 task types:
- **Ingredient** - Find drug ingredients from clinical trials
- **Applicant_Full_Name** - Identify pharmaceutical companies
- **Patent_Expire_Date** - Find patent expiration dates
- **Exclusivity_Date** - Find FDA exclusivity dates
- **Open_on_Approval** - Get stock prices on approval dates

## The `browse` Tool

This template uses **Crawl4AI** to provide a real web browsing tool:

```python
@tool
def browse(url: str) -> str:
    """Browse a URL and return its content as markdown."""
```

The teacher model calls `browse()` with URLs like:
- `https://clinicaltrials.gov/study/NCT01234567`
- `https://www.accessdata.fda.gov/scripts/cder/daf/`

And gets back **real page content** as LLM-friendly markdown.

## Options

```bash
# With real web browsing (recommended)
python generate_distillation_data.py prompts.jsonl \
  --tools tools.py \
  --model openai/gpt-4o \
  --output train.jsonl \
  --concurrency 5

# Without tools (teacher answers from training knowledge only)
python generate_distillation_data.py prompts.jsonl \
  --tools none \
  --model openai/gpt-4o \
  --output train.jsonl

# With custom system prompt
python generate_distillation_data.py prompts.jsonl \
  --tools tools.py \
  --system "You are a medical research assistant. Use the browse tool to find accurate information." \
  --output train.jsonl
```

## Files

| File | Purpose |
|------|---------|
| `convert_dataset.py` | Download MedBrowseComp → prompts.jsonl |
| `generate_distillation_data.py` | Generate teacher completions |
| `tools.py` | Crawl4AI `browse` tool for real web content |
| `config.yml` | SFT training config |
