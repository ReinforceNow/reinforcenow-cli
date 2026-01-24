# Browserbase Playwright Agent

Train an RL agent to answer factual questions using Playwright MCP + Browserbase cloud browsers.

## Overview

This template trains a model to:
1. Navigate to search engines and websites via Browserbase cloud browsers
2. Use standard Playwright commands (click, type, snapshot) to interact with pages
3. Extract information and provide accurate answers

**Key difference from `stagehand-agent`**: Uses direct Playwright commands instead of Stagehand:
- Standard browser automation commands (click, type, navigate)
- Lower latency per action (no AI interpretation for actions)
- More precise control over browser interactions

## Setup

### 1. Get API Keys

1. **Browserbase**: Sign up at [browserbase.com](https://browserbase.com), get API key + project ID
2. **OpenAI**: Get API key from [platform.openai.com](https://platform.openai.com/api-keys) (for LLM judge reward)

### 2. Configure Environment

Copy `example.env` to `.env` and fill in your keys:

```bash
cp example.env .env
```

```env
# Browserbase credentials (https://browserbase.com)
BROWSERBASE_API_KEY=bb_live_xxx
BROWSERBASE_PROJECT_ID=xxx

# OpenAI API key (required for LLM judge reward)
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
| `config.yml` | Training config |
| `train.jsonl` | 500 SimpleQA questions |
| `rewards.py` | used_browser (precondition) + accuracy (LLM judge) |
| `Dockerfile.browserbase` | Playwright MCP + Browserbase CDP with session cleanup |
| `example.env` | Environment variables template |

## Browser Tools (via Playwright MCP)

The Playwright MCP server provides these tools:

| Tool | Description |
|------|-------------|
| `browser_navigate(url)` | Navigate to a URL |
| `browser_click(element, ref)` | Click an element by reference number |
| `browser_type(element, ref, text)` | Type text into an input field |
| `browser_snapshot()` | Get accessibility tree of current page |
| `browser_scroll_down()` | Scroll down the page |
| `browser_scroll_up()` | Scroll up the page |
| `browser_go_back()` | Go back in history |
| `browser_go_forward()` | Go forward in history |

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

## Session Cleanup

The Dockerfile includes automatic session cleanup:
- On container stop (SIGTERM/SIGINT), the Browserbase session is released via API
- This prevents orphaned sessions from counting against your concurrent session limit

## Stagehand vs Playwright

| Feature | Stagehand | Playwright |
|---------|-----------|------------|
| Commands | Natural language ("click login") | Direct selectors (click ref=5) |
| AI Layer | Yes (uses OPENAI_API_KEY) | No |
| Latency | Higher (AI interprets actions) | Lower (direct commands) |
| Flexibility | More forgiving | More precise |

Choose **Stagehand** if you want natural language actions.
Choose **Playwright** if you want lower latency and more precise control.

## Pricing

Browserbase pricing:
- **Developer**: $20/month for 100 browser-hours (~1,200-3,000 rollouts)
- **Startup**: $99/month for 500 browser-hours (~6,000-15,000 rollouts)
