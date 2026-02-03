# RL with Tools Template

Train an agent that can use tools (function calling) with reinforcement learning.

## Quick Start

```bash
# 1. Prepare train.jsonl, rewards.py, and tools.py
# 2. Test locally
rnow test -n 3 --verbose

# 3. Train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | RL training config with multi-turn settings |
| `train.jsonl` | Training prompts with tools and rewards |
| `rewards.py` | Reward function definitions |
| `tools.py` | Tool function definitions |
| `requirements.txt` | Python dependencies |

## train.jsonl Format

```json
{"messages": [{"role": "user", "content": "Search for AI news"}], "rewards": ["relevance"], "tools": ["web_search"]}
```

The `tools` field filters which tools are available. Omit to use all tools from tools.py.

## tools.py Example

```python
from rnow.core import tool

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return "Search results..."
```

## Config for Multi-Turn

```yaml
rollout:
  max_turns: 5  # Allow multiple tool calls
  termination_policy: last_tool  # End when assistant responds without tool
```
