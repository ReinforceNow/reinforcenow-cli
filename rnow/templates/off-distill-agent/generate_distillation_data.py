#!/usr/bin/env python3
"""
Generate off-policy agentic distillation data using OpenRouter.

Runs concurrent agentic rollouts until the model responds without tool calls.
Uses adaptive concurrency - starts high and reduces on rate limit errors.

Usage:
    python generate_distillation_data.py prompts.jsonl --model openai/gpt-4o -n 100
    python generate_distillation_data.py prompts.jsonl --tools tools.py --output train.jsonl
"""

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============== Tool Loading ==============


def load_tools(tools_path: str) -> dict:
    """Load tools from a Python file. Returns {name: {func, schema, desc}}."""
    if not Path(tools_path).exists():
        return {}

    # Mock the @tool decorator
    registry = {}

    class MockTool:
        def __call__(self, fn=None, **kwargs):
            def decorator(func):
                schema = {"type": "object", "properties": {}, "required": []}
                hints = getattr(func, "__annotations__", {})
                sig = inspect.signature(func)

                for name, param in sig.parameters.items():
                    if name in ("self", "cls"):
                        continue
                    prop = {"type": "string"}
                    if name in hints:
                        t = hints[name]
                        if t is int:
                            prop = {"type": "integer"}
                        elif t is float:
                            prop = {"type": "number"}
                        elif t is bool:
                            prop = {"type": "boolean"}
                        elif t is list:
                            prop = {"type": "array"}
                        elif t is dict:
                            prop = {"type": "object"}
                    schema["properties"][name] = prop
                    if param.default is inspect.Parameter.empty:
                        schema["required"].append(name)

                registry[func.__name__] = {
                    "func": func,
                    "schema": schema,
                    "desc": (func.__doc__ or "").strip().split("\n")[0],
                }
                return func

            return decorator(fn) if fn else decorator

    # Inject mock
    mock = MockTool()
    sys.modules["rnow"] = type("M", (), {"core": type("M", (), {"tool": mock})()})()
    sys.modules["rnow.core"] = type("M", (), {"tool": mock})()
    sys.modules["rnow.core.tool"] = type("M", (), {"tool": mock})()

    spec = importlib.util.spec_from_file_location("tools", tools_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return registry


def tools_to_openai(tools: dict) -> list:
    """Convert tools to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {"name": n, "description": t["desc"], "parameters": t["schema"]},
        }
        for n, t in tools.items()
    ]


def execute_tool(tools: dict, name: str, args: dict) -> str:
    """Execute a tool and return result as string."""
    if name not in tools:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = tools[name]["func"](**args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============== Rollout Generation ==============


async def run_rollout(
    client: httpx.AsyncClient,
    prompt: dict,
    tools: dict,
    model: str,
    system: str,
    max_tokens: int,
    max_turns: int,
    temperature: float,
    api_key: str,
) -> dict | None:
    """Run single agentic rollout until assistant responds without tools."""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt["prompt"]})

    openai_tools = tools_to_openai(tools) if tools else None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for _turn in range(max_turns):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if openai_tools:
            payload["tools"] = openai_tools

        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise  # Re-raise rate limit errors for adaptive handling
            print(f"\nAPI error: {e}")
            return None
        except Exception as e:
            print(f"\nAPI error: {e}")
            return None

        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls", [])

        # Build assistant message - always include content field (empty string if null)
        asst = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            asst["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        messages.append(asst)

        # No tool calls = done
        if not tool_calls:
            break

        # Execute tools
        for tc in tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"])
            except:
                args = {}
            result = execute_tool(tools, tc["function"]["name"], args)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # Build output entry
    entry = {"messages": messages, "metadata": {"teacher": model, "turns": _turn + 1}}
    if prompt.get("metadata"):
        entry["metadata"].update(prompt["metadata"])
    if tools:
        entry["tools"] = list(tools.keys())

    return entry


async def generate_all(
    prompts: list[dict],
    tools: dict,
    output: str,
    model: str,
    system: str,
    max_tokens: int,
    max_turns: int,
    temperature: float,
    initial_concurrency: int,
    api_key: str,
):
    """Generate all rollouts with adaptive concurrency."""

    # Resume support
    existing = set()
    if Path(output).exists():
        for line in Path(output).read_text().splitlines():
            try:
                d = json.loads(line)
                for m in d.get("messages", []):
                    if m.get("role") == "user":
                        existing.add(m.get("content", ""))
                        break
            except:
                pass

    remaining = [p for p in prompts if p["prompt"] not in existing]
    if not remaining:
        print("All done!")
        return

    if len(remaining) < len(prompts):
        print(f"Resuming: {len(prompts) - len(remaining)} done, {len(remaining)} remaining")

    concurrency = initial_concurrency
    completed = 0
    failed = 0
    rate_limited = 0

    async with httpx.AsyncClient() as client:
        with open(output, "a") as f:
            pbar = tqdm(total=len(remaining), desc=f"Generating (c={concurrency})")

            i = 0
            while i < len(remaining):
                # Process batch with current concurrency
                batch = remaining[i : i + concurrency]
                tasks = [
                    run_rollout(
                        client, p, tools, model, system, max_tokens, max_turns, temperature, api_key
                    )
                    for p in batch
                ]

                batch_rate_limited = 0
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, httpx.HTTPStatusError) and r.response.status_code == 429:
                        batch_rate_limited += 1
                        rate_limited += 1
                    elif isinstance(r, Exception) or r is None:
                        failed += 1
                    else:
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        completed += 1
                        pbar.update(1)

                # Adaptive concurrency: reduce if we hit rate limits
                if batch_rate_limited > 0:
                    old_concurrency = concurrency
                    concurrency = max(1, concurrency // 2)
                    if concurrency != old_concurrency:
                        pbar.set_description(f"Generating (c={concurrency}, rate limited)")
                        print(
                            f"\n⚠️  Rate limited ({batch_rate_limited}x), reducing concurrency: {old_concurrency} → {concurrency}"
                        )
                        # Re-add failed items to remaining
                        remaining = remaining[i:] + [
                            p
                            for p, r in zip(batch, results, strict=False)
                            if isinstance(r, httpx.HTTPStatusError)
                            and r.response.status_code == 429
                        ]
                        i = 0
                        await asyncio.sleep(2)  # Brief pause after rate limit
                        continue

                i += len(batch)

            pbar.close()

    print(f"\nCompleted: {completed}, Failed: {failed}, Rate limited: {rate_limited}")


# ============== Main ==============


def load_prompts(path: str) -> list[dict]:
    """Load prompts from txt or jsonl."""
    prompts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict) and "prompt" in d:
                prompts.append(d)
            else:
                prompts.append({"prompt": line})
        except:
            prompts.append({"prompt": line})
    return prompts


def main():
    p = argparse.ArgumentParser(description="Generate off-policy agentic distillation data")
    p.add_argument("input", help="Input prompts file (txt or jsonl)")
    p.add_argument("-o", "--output", default="train.jsonl", help="Output file")
    p.add_argument("-t", "--tools", default="tools.py", help="Tools file (use 'none' to disable)")
    p.add_argument("-m", "--model", default="openai/gpt-4o-mini", help="Teacher model")
    p.add_argument("-s", "--system", default=None, help="System prompt")
    p.add_argument("-n", "--num", type=int, help="Max prompts to process")
    p.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=20,
        help="Initial concurrent requests (adapts down on rate limits)",
    )
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    p.add_argument("--max-turns", type=int, default=10, help="Max turns per rollout")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    args = p.parse_args()

    # Load API key
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Set OPENROUTER_API_KEY in .env or environment")
        sys.exit(1)

    # Load tools
    tools = {} if args.tools.lower() == "none" else load_tools(args.tools)
    print(f"Tools: {list(tools.keys()) or 'none (direct answers)'}")

    system = args.system
    if system:
        print(f"System: {system[:60]}...")

    # Load prompts
    prompts = load_prompts(args.input)
    if args.num:
        prompts = prompts[: args.num]
    print(f"Prompts: {len(prompts)}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency} (adaptive)")
    print()

    asyncio.run(
        generate_all(
            prompts,
            tools,
            args.output,
            args.model,
            system,
            args.max_tokens,
            args.max_turns,
            args.temperature,
            args.concurrency,
            api_key,
        )
    )


if __name__ == "__main__":
    main()
