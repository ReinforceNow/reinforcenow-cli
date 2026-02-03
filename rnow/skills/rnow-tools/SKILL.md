---
name: rnow-tools
description: Write tool functions for ReinforceNow agent training. Use when creating @tool decorated functions, writing tools.py, or sandbox tools. Triggers on "@tool", "tools.py", "tool function", "function calling", "agent tools", "sandbox".
---

# Writing Tool Functions

Tools allow LLMs to call external functions during training. The model learns when and how to use tools through reinforcement learning.

## Basic Structure

Every tool function must:
1. Be decorated with `@tool`
2. Have type hints on ALL parameters
3. Have a docstring
4. Return JSON-serializable data

```python
from rnow.core.tool import tool

@tool
def my_tool(query: str, limit: int = 10) -> dict:
    """Brief description of what this tool does.

    Args:
        query: What to search for
        limit: Maximum results to return
    """
    return {"results": [...]}
```

## Stateless Tools

For tools that don't modify state (API calls, calculations):

```python
from rnow.core.tool import tool

@tool
def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 3 * 4"
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters"}
        return {"result": eval(expression)}
    except Exception as e:
        return {"error": str(e)}
```

## Stateful Tools (Sandbox)

For tools that modify state (file operations, code execution, installing packages), use `sandbox=True`. This runs the tool in an isolated Docker container where state persists between tool calls within the same rollout.

**Requirements:**
- Add `sandbox=True` to the `@tool` decorator
- Add `"docker": "image:tag"` to train.jsonl entries (see **rnow-train-jsonl** skill)

```python
from rnow.core.tool import tool
import subprocess

@tool(sandbox=True, timeout=120)
def execute_python(code: str) -> dict:
    """Execute Python code in isolated sandbox.

    Args:
        code: Python code to execute
    """
    with open("script.py", "w") as f:
        f.write(code)

    result = subprocess.run(
        ["python", "script.py"],
        capture_output=True,
        text=True,
        timeout=60
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

@tool(sandbox=True)
def write_file(path: str, content: str) -> dict:
    """Write content to a file.

    Args:
        path: Path to write to
        content: Content to write
    """
    with open(path, 'w') as f:
        f.write(content)
    return {"success": True, "path": path}

@tool(sandbox=True)
def read_file(path: str) -> dict:
    """Read contents of a file.

    Args:
        path: Path to the file
    """
    try:
        with open(path, 'r') as f:
            return {"content": f.read()}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
```

## Tool Options

| Option | Default | Description |
|--------|---------|-------------|
| `sandbox` | `False` | Run in isolated Docker container (state persists between calls) |
| `timeout` | `60` | Execution timeout in seconds |

## Config (config.yml)

Configure tool behavior in the `rollout` section:

```yaml
rollout:
  max_turns: 5              # Max tool calls before final response
  termination_policy: last_tool  # End when model responds without tool call
  tool_timeout: 60          # Per-tool execution timeout (seconds)
  max_context_window: 32768      # Max context window in tokens (tool results auto-truncated)
```

For full config options, see the **rnow-config** skill.

For train.jsonl entry format and filtering tools per entry, see the **rnow-train-jsonl** skill.
