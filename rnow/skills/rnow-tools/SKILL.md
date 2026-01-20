---
name: rnow-tools
description: Write tool functions for ReinforceNow agent training. Use when creating @tool decorated functions, writing tools.py, defining function calling tools, or sandbox tools. Triggers on "@tool", "tools.py", "tool function", "function calling", "agent tools".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Writing Tool Functions for ReinforceNow

Tools allow LLMs to call external functions during training. The model learns when and how to use tools through reinforcement learning.

## Basic Structure

Every tool function must:
1. Be decorated with `@tool`
2. Have type hints on ALL parameters
3. Have a non-empty docstring
4. Return JSON-serializable data

```python
from rnow.core.tool import tool

@tool
def my_tool(query: str, limit: int = 10) -> dict:
    """Brief description of what this tool does.

    Args:
        query: What to search for
        limit: Maximum results to return

    Returns:
        Dict with results
    """
    return {"results": [...]}
```

## Type Requirements

### Supported Parameter Types

| Type | Example |
|------|---------|
| `str` | `query: str` |
| `int` | `count: int` |
| `float` | `threshold: float` |
| `bool` | `verbose: bool` |
| `list` | `items: list` |
| `list[T]` | `ids: list[int]` |
| `dict` | `options: dict` |
| `dict[K, V]` | `mapping: dict[str, int]` |
| `Optional[T]` | `name: Optional[str]` |
| `T | None` | `value: str | None` |
| `Literal[...]` | `mode: Literal["fast", "slow"]` |
| `Union[A, B]` | `data: Union[str, int]` |

### Return Types

Must be JSON-serializable:
- `str`, `int`, `float`, `bool`
- `list`, `dict`
- `None`
- Nested combinations of above

## Tool Patterns

### 1. Web Search

```python
import requests
from rnow.core.tool import tool

@tool
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results with title and snippet
    """
    resp = requests.get(
        "https://api.search.example.com/search",
        params={"q": query, "limit": max_results},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()["results"]
```

### 2. Wikipedia Search

```python
import requests
from bs4 import BeautifulSoup
from rnow.core.tool import tool

@tool
def wikipedia_search(query: str) -> list:
    """Search Wikipedia and return article summaries.

    Args:
        query: Topic to search for

    Returns:
        List of articles with title, link, and snippet
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5
            },
            headers={"User-Agent": "ReinforceNow/1.0"},
            timeout=10
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return [{"error": str(e)}]

    data = resp.json()
    results = []
    for item in data.get("query", {}).get("search", []):
        snippet = BeautifulSoup(
            item.get("snippet", ""), "html.parser"
        ).get_text()
        title = item.get("title", "")
        results.append({
            "title": title,
            "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "snippet": snippet[:200]
        })
    return results
```

### 3. Calculator

```python
from rnow.core.tool import tool

@tool
def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 3 * 4"

    Returns:
        Dict with result or error
    """
    try:
        # Safe evaluation (only math operations)
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters in expression"}
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

### 4. Database Query

```python
from rnow.core.tool import tool

@tool
def query_database(sql: str, database: str = "main") -> dict:
    """Execute a read-only SQL query.

    Args:
        sql: SQL SELECT query
        database: Database name

    Returns:
        Query results as list of dicts
    """
    # Validate it's a SELECT query
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries allowed"}

    # Execute query (implementation depends on your setup)
    results = execute_sql(sql, database)
    return {"rows": results, "count": len(results)}
```

### 5. File Operations

```python
from rnow.core.tool import tool

@tool
def read_file(path: str) -> dict:
    """Read contents of a file.

    Args:
        path: Path to the file

    Returns:
        File contents or error
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
        return {"content": content, "size": len(content)}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except Exception as e:
        return {"error": str(e)}

@tool
def write_file(path: str, content: str) -> dict:
    """Write content to a file.

    Args:
        path: Path to write to
        content: Content to write

    Returns:
        Success status
    """
    try:
        with open(path, 'w') as f:
            f.write(content)
        return {"success": True, "path": path}
    except Exception as e:
        return {"error": str(e)}
```

### 6. API Client

```python
import requests
from typing import Optional
from rnow.core.tool import tool

@tool
def call_api(
    endpoint: str,
    method: Literal["GET", "POST"] = "GET",
    data: Optional[dict] = None
) -> dict:
    """Make an API request.

    Args:
        endpoint: API endpoint URL
        method: HTTP method
        data: Request body for POST

    Returns:
        API response
    """
    try:
        if method == "GET":
            resp = requests.get(endpoint, timeout=30)
        else:
            resp = requests.post(endpoint, json=data, timeout=30)
        resp.raise_for_status()
        return {"status": resp.status_code, "data": resp.json()}
    except Exception as e:
        return {"error": str(e)}
```

## Sandbox Tools

Use `sandbox=True` when tools need isolated execution:
- Code execution
- File system operations
- Environment modifications

**IMPORTANT**: Entries using sandbox tools MUST have `docker` field in train.jsonl.

```python
@tool(sandbox=True, timeout=120)
def execute_python(code: str) -> dict:
    """Execute Python code in isolated sandbox.

    Args:
        code: Python code to execute

    Returns:
        Execution output or error
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        filepath = f.name

    try:
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out"}

@tool(sandbox=True)
def run_bash(command: str) -> dict:
    """Execute a bash command in sandbox.

    Args:
        command: Bash command to run

    Returns:
        Command output
    """
    import subprocess

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

@tool(sandbox=True)
def install_package(package: str) -> dict:
    """Install a Python package in sandbox.

    Args:
        package: Package name to install

    Returns:
        Installation result
    """
    import subprocess

    result = subprocess.run(
        ["pip", "install", package],
        capture_output=True,
        text=True,
        timeout=120
    )
    return {
        "success": result.returncode == 0,
        "output": result.stdout + result.stderr
    }
```

train.jsonl entry for sandbox tools:
```json
{
  "messages": [{"role": "user", "content": "Write and run a Python script"}],
  "rewards": ["code_works"],
  "tools": ["execute_python"],
  "docker": "python:3.11-slim"
}
```

## Tool Options

| Option | Default | Description |
|--------|---------|-------------|
| `sandbox` | `False` | Run in isolated Docker container |
| `timeout` | `60` | Execution timeout in seconds |

```python
@tool(sandbox=True, timeout=300)
def long_running_task(data: str) -> dict:
    """Task that may take up to 5 minutes."""
    # ...
```

## Filtering Tools Per Entry

In train.jsonl, use `tools` field to limit which tools are available:

```json
{"messages": [...], "tools": ["search", "calculator"]}
```

If `tools` is omitted, ALL defined tools are available.

## Config for Multi-Turn Agents

In config.yml:

```yaml
rollout:
  max_turns: 5              # Max tool calls before final response
  termination_policy: last_tool  # End when no tool call
  tool_timeout: 60          # Per-tool timeout
  max_tool_response_chars: 4000  # Truncate long responses
```

### Termination Policies

| Policy | Behavior |
|--------|----------|
| `last_tool` | Episode ends when model responds without tool call |
| `max_turns` | Episode always runs for exactly max_turns |

## Async Tools

Both sync and async work:

```python
# Sync
@tool
def sync_tool(query: str) -> dict:
    return {"result": process(query)}

# Async (for I/O-bound operations)
@tool
async def async_tool(query: str) -> dict:
    result = await async_process(query)
    return {"result": result}
```

## Common Mistakes

### Wrong: Missing type hints
```python
@tool
def bad_tool(query):  # ERROR: No type hint
    return {"result": query}
```

### Wrong: Missing docstring
```python
@tool
def bad_tool(query: str) -> dict:
    # ERROR: No docstring
    return {"result": query}
```

### Wrong: Using *args or **kwargs
```python
@tool
def bad_tool(*args, **kwargs) -> dict:  # ERROR
    return {}
```

### Wrong: Non-JSON-serializable return
```python
@tool
def bad_tool(query: str) -> set:  # ERROR: set not JSON-serializable
    return {1, 2, 3}
```

### Wrong: sandbox=True without docker field
```python
@tool(sandbox=True)
def run_code(code: str) -> dict:
    # ERROR if train.jsonl entry lacks "docker" field
    exec(code)
    return {"success": True}
```

### Right: Complete tool definition
```python
@tool
def good_tool(
    query: str,
    limit: int = 10,
    include_metadata: bool = False
) -> dict:
    """Search for items matching query.

    Args:
        query: Search query string
        limit: Maximum results (default 10)
        include_metadata: Include extra info

    Returns:
        Dict with results list
    """
    results = perform_search(query, limit)
    if include_metadata:
        results = add_metadata(results)
    return {"results": results, "count": len(results)}
```

## Testing Tools Locally

```bash
rnow test -n 3 --verbose --with-tools
```

This runs rollouts with tools enabled and shows tool calls/responses.
