# RL Browser Agent

Train an RL agent to answer factual questions using web browsing with Playwright MCP.

## Overview

This template trains a model to:
1. Navigate to search engines and websites
2. Browse web pages to find information
3. Extract and provide accurate answers

The agent learns through RL rewards - getting rewarded for correct answers.

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
| `rewards.py` | used_browser (precondition) + accuracy |
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

### used_browser (precondition)
Gate reward - must use at least one browser tool to get any reward.

### accuracy (0.0 or 1.0)
Checks if "Final Answer: <answer>" matches expected answer using Jaro-Winkler similarity (>90% required).

## train.jsonl Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"}
  ],
  "rewards": ["used_browser", "accuracy"],
  "docker": "local/playwright",
  "metadata": {"expected_answer": "Michio Sugeno"}
}
```

## Converting SimpleQA Dataset

```python
from datasets import load_dataset
import json

ds = load_dataset("openai/SimpleQA", split="train")

system_prompt = """You are a research assistant. Answer questions by finding proof on the web using browser tools.

After finding evidence, provide your answer in this format:
Final Answer: <your answer>"""

with open("train.jsonl", "w") as f:
    for row in ds:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["problem"]},
            ],
            "rewards": ["used_browser", "accuracy"],
            "docker": "local/playwright",
            "metadata": {"expected_answer": row["answer"]},
        }
        f.write(json.dumps(entry) + "\n")
```

## Config Options

```yaml
dataset_type: rl  # RL training with rewards

rollout:
  max_turns: 25          # Allow multiple browse iterations
  max_tokens: 4096       # Token limit per response
  tool_timeout: 120      # Seconds to wait for browser tools
  mcp_url: localhost:8931  # Playwright MCP server port

algorithm:
  kl_penalty_coef: 0.01  # Low KL penalty for exploration
```
