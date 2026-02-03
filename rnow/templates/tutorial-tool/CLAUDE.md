# Tutorial: Writing Tool Functions

Learn how to write tool functions for ReinforceNow agent training.

## Quick Start

```bash
# Test tools with the model
rnow test -n 3 --verbose

# Once working, train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Training config with multi-turn |
| `train.jsonl` | Prompts that require tool use |
| `rewards.py` | Reward functions |
| `tools.py` | Tool definitions |
| `requirements.txt` | Dependencies |

## Tool Function Signature

```python
from rnow.core import tool

@tool
def my_tool(param: str, count: int = 10) -> str:
    """Description shown to the model.

    Args:
        param: What this parameter does
        count: Optional count (default 10)
    """
    # Implementation
    return "Result string"
```

## Tool Types

### Basic Tool
```python
@tool
def calculator(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)
```

### Sandbox Tool (isolated execution)
```python
@tool(sandbox=True)
def run_python(code: str) -> str:
    """Run Python code in sandbox."""
    exec(code)
    return "Success"
```

Sandbox tools require `"docker": "python:3.11-slim"` in train.jsonl entries.

## Config for Tools

```yaml
rollout:
  max_turns: 5  # Multiple tool calls allowed
  termination_policy: last_tool  # End when no tool called
  tool_timeout: 60  # Seconds per tool call
```
