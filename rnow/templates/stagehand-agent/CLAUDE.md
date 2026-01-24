# Stagehand Agent

Train an RL agent to answer factual questions using Stagehand + Browserbase.

## Overview

This template trains a model to:
1. Navigate to search engines and websites via Browserbase cloud browsers
2. Use natural language actions (Stagehand) to interact with pages
3. Extract information and provide accurate answers

**Key difference from `rl-browser`**: Uses Browserbase cloud browsers instead of local Playwright:
- Managed browser infrastructure (no local browser)
- Stagehand natural language actions (`act("click the login button")`)
- Built-in proxy and anti-detection

## Setup

### 1. Get API Keys

1. **Browserbase**: Sign up at [browserbase.com](https://browserbase.com), get API key + project ID
2. **OpenAI**: Get API key from [platform.openai.com](https://platform.openai.com/api-keys)

### 2. Configure Environment

Copy `example.env` to `.env` and fill in your keys:

```bash
cp example.env .env
```

```env
# Browserbase credentials (https://browserbase.com)
BROWSERBASE_API_KEY=bb_live_xxx
BROWSERBASE_PROJECT_ID=xxx

# OpenAI API key (required)
# Used by both:
# 1. Stagehand - for natural language browser actions
# 2. LLM judge reward - for evaluating answer accuracy
OPENAI_API_KEY=sk-xxx
```

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
| `config.yml` | Training config with `mcp_url: localhost:8931` |
| `train.jsonl` | 500 SimpleQA questions with docker field |
| `rewards.py` | used_browser (precondition) + accuracy (LLM judge) |
| `Dockerfile.browserbase` | Runs official `@browserbasehq/mcp-server-browserbase` |
| `example.env` | Environment variables template |

## Browser Tools (via MCP)

The Browserbase MCP server provides Stagehand-powered tools:

| Tool | Description |
|------|-------------|
| `browserbase_stagehand_navigate(url)` | Navigate to a URL |
| `browserbase_stagehand_act(action)` | Natural language action (e.g., "click the search button") |
| `browserbase_stagehand_extract()` | Extract all text from current page |
| `browserbase_stagehand_observe(instruction)` | Find actionable elements |
| `browserbase_screenshot()` | Take a PNG screenshot |
| `browserbase_stagehand_get_url()` | Get current URL |

## Rewards

### used_browser (precondition)
Gate reward - must use at least one browser tool to get any reward.

### accuracy (LLM judge)
Uses GPT to evaluate if the model's response semantically matches the expected answer.

Based on **LLM-as-a-Judge best practices**:
- Binary scoring (0 or 1)
- Clear criteria definition
- Chain-of-thought reasoning
- Handles semantic equivalence (not just exact match)

The LLM judge allows flexible matching - "Michio Sugeno" matches "Sugeno, Michio", and "2010" matches "in 2010".

## train.jsonl Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"}
  ],
  "rewards": ["used_browser", "accuracy"],
  "docker": "local/browserbase",
  "metadata": {"expected_answer": "Michio Sugeno"}
}
```

## Pricing

Browserbase pricing:
- **Developer**: $20/month for 100 browser-hours (~1,200-3,000 rollouts)
- **Startup**: $99/month for 500 browser-hours (~6,000-15,000 rollouts)
