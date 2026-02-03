# MCP Tavily Search Agent

Train an agent using Tavily search via MCP (Model Context Protocol).

## Setup

1. Get a Tavily API key from https://tavily.com
2. Add to `.env`:
```bash
TAVILY_API_KEY=tvly-xxx
```

## Quick Start

```bash
# Test with MCP tools
rnow test -n 3 --verbose

# Train
rnow run
```

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Config with MCP URL |
| `train.jsonl` | QA prompts |
| `rewards.py` | Accuracy reward |
| `requirements.txt` | Dependencies |

## How MCP Works

The `mcp_url` in config.yml connects to an external MCP server that provides tools:

```yaml
rollout:
  mcp_url: "https://mcp.tavily.com/search"  # External MCP
  max_turns: 3
```

The model receives tool schemas from MCP and can call them during rollouts.

## MCP vs tools.py

| Feature | tools.py | MCP |
|---------|----------|-----|
| Where tools run | Sidecar/Sandbox | External server |
| Setup | Write Python | Connect to URL |
| Use case | Custom logic | Third-party services |

You can use both together - tools from both sources are merged.
