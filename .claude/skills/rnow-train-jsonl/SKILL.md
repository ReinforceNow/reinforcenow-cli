---
name: rnow-train-jsonl
description: Format train.jsonl training data for ReinforceNow. Use when creating train.jsonl, formatting training entries, using tools/rewards per entry, or setting up sandbox/docker. Triggers on "train.jsonl", "training data", "docker", "sandbox", "entry format".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# train.jsonl Format

One JSON object per line. Each entry is a training example.

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | Yes | Conversation array |
| `rewards` | RL only | List of reward function names |
| `metadata` | No | Data accessible via `args.metadata` in rewards |
| `variables` | No | Template variables via `args.variables` |
| `tools` | No | Filter which tools are available for this entry |
| `docker` | If sandbox | Docker image for sandbox execution |
| `docker_env` | No | Environment variables for sandbox |
| `docker_cmd` | No | Custom entrypoint command |

## Message Roles

| Role | Description |
|------|-------------|
| `system` | System instructions (optional, must be first) |
| `user` | User message (at least one required) |
| `assistant` | Assistant response (for multi-turn context) |
| `tool` | Tool call result (for tool use context) |

## Basic Examples

### RL Entry

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}], "rewards": ["accuracy"], "metadata": {"answer": "4"}}
```

### SFT Entry

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

### SFT with Tool Calls (Agentic Distillation)

SFT supports training on conversations with tool calls (e.g., from teacher model distillation):

```json
{
  "messages": [
    {"role": "user", "content": "Find the weather in Paris"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "72°F, sunny"},
    {"role": "assistant", "content": "The weather in Paris is 72°F and sunny."}
  ]
}
```

**Tool call format** (OpenAI-compatible):
```json
{
  "id": "call_xxx",
  "type": "function",
  "function": {
    "name": "tool_name",
    "arguments": "{\"arg\": \"value\"}"
  }
}
```

**Notes:**
- `arguments` must be a JSON string, not an object
- `content` can be empty string `""` when assistant makes tool calls
- Tool results use `role: "tool"` with matching `tool_call_id`
- Works with all model renderers (Qwen3, DeepSeek, Kimi, etc.)

### With System Prompt

```json
{"messages": [{"role": "system", "content": "You are a math tutor"}, {"role": "user", "content": "Explain fractions"}], "rewards": ["quality"]}
```

## Using Tools

Filter which tools are available for a specific entry with the `tools` field:

```json
{"messages": [{"role": "user", "content": "Search for AI news"}], "rewards": ["relevance"], "tools": ["web_search"]}
```

If `tools` is omitted, ALL defined tools in tools.py are available.

For writing tool functions, see the **rnow-tools** skill.

## Sandbox Entries

For entries that need isolated execution (code execution, file operations), use the `docker` field. This spawns a Modal sandbox where state persists between tool calls within the same rollout.

**Required when**: Any reward or tool uses `sandbox=True`.

### Basic Sandbox

```json
{
  "messages": [{"role": "user", "content": "Write and run a Python script"}],
  "rewards": ["code_runs", "output_correct"],
  "tools": ["execute_python"],
  "docker": "python:3.11-slim"
}
```

### Custom Docker Image

```json
{
  "messages": [{"role": "user", "content": "Analyze the data"}],
  "rewards": ["accuracy"],
  "docker": "myorg/custom-image:latest",
  "docker_env": {"DEBUG": "true", "DATA_PATH": "/data"},
  "docker_cmd": ["python", "setup.py", "--init"]
}
```

### Building Custom Images

**CRITICAL**: Docker images must be built for `linux/amd64`:

```bash
# Correct - Modal compatible
docker build --platform linux/amd64 -t myorg/image:latest .
docker push myorg/image:latest

# Wrong - will fail on x86_64 servers
docker build -t myorg/image:latest .
```

Modal runs on x86_64 Linux servers. Images built on ARM Macs without `--platform linux/amd64` will fail.

## Multi-Turn Context

Provide conversation history for multi-turn training:

```json
{
  "messages": [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "What's its population?"}
  ],
  "rewards": ["accuracy"],
  "metadata": {"answer": "2.1 million"}
}
```

## Validation Rules

1. **Rewards must exist** - Names in `rewards` must match `@reward` functions in rewards.py
2. **Tools must exist** - Names in `tools` must match `@tool` functions in tools.py
3. **sandbox=True requires docker** - If any reward/tool uses `sandbox=True`, the entry needs a `docker` field
4. **Messages format** - Must have at least one `user` message; `system` must be first if present

## Related Skills

- **rnow-tools** - Writing tool functions (@tool decorator)
- **rnow-rewards** - Writing reward functions (@reward decorator)
- **rnow-config** - config.yml settings and HuggingFace dataset conversion
