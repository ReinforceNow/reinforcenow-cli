# RL Browser Agent

Train an RL agent to answer factual questions using web browsing with Crawl4AI.

## Overview

This template trains a model to:
1. Search Wikipedia for relevant pages
2. Browse web pages to read content
3. Extract and provide accurate answers

The agent learns through RL rewards - getting rewarded for correct answers and for using the browse tool.

## Docker Sandbox

This template uses **local Dockerfiles** for Docker sandboxes:
- `tools.py` uses `@tool(sandbox=True)` to run in isolated containers
- `train.jsonl` entries have `"docker": "local/crawl4ai"` field
- `Dockerfile.crawl4ai` in project root starts the crawl4ai HTTP server
- Modal builds the image automatically - no need to push to a registry

## Quick Start

```bash
# Test locally with a few samples
rnow test -n 3 --verbose

# Run full training
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Training configuration |
| `train.jsonl` | Training data with docker field |
| `tools.py` | browse + search_wikipedia tools with sandbox=True |
| `rewards.py` | accuracy + used_browse rewards |
| `requirements.txt` | Sidecar dependencies (jellyfish) |
| `Dockerfile.crawl4ai` | Local Dockerfile that starts crawl4ai server |

## Tools

### browse(url: str) -> str
Fetches a web page using Crawl4AI and returns content as markdown.
- Runs in Docker sandbox (built from `Dockerfile.crawl4ai`)
- Calls crawl4ai HTTP API at `http://localhost:11235/crawl`
- Truncates output to 8000 characters

### search_wikipedia(query: str) -> str
Searches Wikipedia and returns titles + snippets.
- Runs in Docker sandbox
- Returns up to 5 results

## Rewards

### accuracy (0.0 or 1.0)
Checks if "Final Answer: <answer>" matches expected answer using Jaro-Winkler similarity (>90% required).

### used_browse (0.0 or 0.2)
Small reward for using the browse tool - encourages research over guessing.

## train.jsonl Format

Each entry includes:
- `messages`: System prompt + user question
- `tools`: List of tool names to make available
- `rewards`: List of reward function names
- `docker`: Docker image for sandbox tools
- `metadata`: Expected answer for reward computation

```json
{
  "messages": [...],
  "tools": ["browse", "search_wikipedia"],
  "rewards": ["accuracy", "used_browse"],
  "docker": "local/crawl4ai",
  "metadata": {"expected_answer": "..."}
}
```

## Converting SimpleQA Dataset

To use the full SimpleQA dataset:

```python
from datasets import load_dataset
import json

ds = load_dataset("openai/SimpleQA", split="train")

system_prompt = """You are a research assistant that answers factual questions by browsing the web.

You have access to these tools:
- browse(url): Fetch a web page and return its content as markdown
- search_wikipedia(query): Search Wikipedia for information

Strategy:
1. Use search_wikipedia to find relevant pages
2. Use browse to read the actual page content
3. Extract the answer from the page
4. Provide your final answer

After your reasoning, provide your final answer in this exact format:
Final Answer: <your answer>"""

with open("train.jsonl", "w") as f:
    for row in ds:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["problem"]},
            ],
            "tools": ["browse", "search_wikipedia"],
            "rewards": ["accuracy", "used_browse"],
            "docker": "local/crawl4ai",
            "metadata": {"expected_answer": row["answer"]},
        }
        f.write(json.dumps(entry) + "\n")
```

## Config Options

```yaml
dataset_type: rl  # RL training with rewards

rollout:
  max_turns: 4        # Allow multiple browse iterations
  max_tokens: 4096    # Token limit per response
  tool_timeout: 120   # Seconds to wait for browse tool

algorithm:
  kl_penalty_coef: 0.01  # Low KL penalty for exploration
```
