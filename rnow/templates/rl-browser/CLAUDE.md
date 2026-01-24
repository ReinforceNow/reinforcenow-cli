# RL Browser Agent

Train an RL agent to answer factual questions using web browsing with Playwright MCP.

## Overview

This template trains a model to:
1. Navigate to search engines and websites
2. Browse web pages to find information
3. Extract and provide accurate answers

The agent learns through RL rewards - getting rewarded for correct answers and for using browser tools.

## Playwright MCP

This template uses **Playwright MCP** for real browser automation:
- Browser tools (navigate, click, type, screenshot) are provided via MCP
- `Dockerfile.playwright` runs the MCP server inside the sandbox
- No `tools.py` needed - MCP provides all browser tools automatically
- `mcp_url: localhost:8931` connects to the MCP server

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
| `config.yml` | Training configuration with `mcp_url` |
| `train.jsonl` | SimpleQA questions with docker field |
| `rewards.py` | accuracy + used_browse rewards |
| `requirements.txt` | Sidecar dependencies (jellyfish) |
| `Dockerfile.playwright` | Playwright MCP server image |

## Browser Tools (via MCP)

The Playwright MCP server provides these tools:
- `browser_navigate(url)` - Navigate to a URL
- `browser_click(element)` - Click an element
- `browser_type(element, text)` - Type text into an element
- `browser_snapshot()` - Get page content/screenshot
- `browser_scroll(direction)` - Scroll the page

## Rewards

### accuracy (0.0 or 1.0)
Checks if "Final Answer: <answer>" matches expected answer using Jaro-Winkler similarity (>90% required).

### used_browse (0.0 or 0.2)
Small reward for using browser tools - encourages research over guessing.

## train.jsonl Format

Each entry includes:
- `messages`: System prompt + user question
- `rewards`: List of reward function names
- `docker`: Docker image for Playwright MCP (`local/playwright`)
- `metadata`: Expected answer for reward computation

```json
{
  "messages": [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"}
  ],
  "rewards": ["accuracy", "used_browse"],
  "docker": "local/playwright",
  "metadata": {"expected_answer": "Michio Sugeno"}
}
```

## Converting SimpleQA Dataset

To use the full SimpleQA dataset:

```python
from datasets import load_dataset
import json

ds = load_dataset("openai/SimpleQA", split="train")

system_prompt = """You are a research assistant that answers factual questions by browsing the web.

You have browser tools to navigate and interact with web pages. Use them to find information.

Strategy:
1. Navigate to a search engine or relevant website
2. Browse pages to find the answer
3. Extract the answer from the page content

After your reasoning, provide your final answer in this exact format:
Final Answer: <your answer>"""

with open("train.jsonl", "w") as f:
    for row in ds:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["problem"]},
            ],
            "rewards": ["accuracy", "used_browse"],
            "docker": "local/playwright",
            "metadata": {"expected_answer": row["answer"]},
        }
        f.write(json.dumps(entry) + "\n")
```

## Config Options

```yaml
dataset_type: rl  # RL training with rewards

rollout:
  max_turns: 4           # Allow multiple browse iterations
  max_tokens: 4096       # Token limit per response
  max_context_window: 32768  # Context window in tokens
  tool_timeout: 120      # Seconds to wait for browser tools
  mcp_url: localhost:8931  # Playwright MCP server port

algorithm:
  kl_penalty_coef: 0.01  # Low KL penalty for exploration
```
