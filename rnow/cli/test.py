# rnow/cli/test.py
"""
Test command for running RL rollouts locally.

Requires authentication for billing.
"""

import asyncio
import itertools
import json
import random
import re
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from string import Template

import click
import httpx
import yaml

# Global flag for graceful shutdown
_shutdown_requested = False

from rnow.cli.auth import get_auth_headers
from rnow.cli.commands import get_thinking_mode_display

# ReinforceNow teal: #14B8A6 as RGB tuple for click.style()
TEAL_RGB = (20, 184, 166)


class Spinner:
    """Simple spinner for CLI feedback with dynamic status updates."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = ""):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def update(self, message: str):
        """Update the spinner message."""
        with self._lock:
            self.message = message

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set() or _shutdown_requested:
                break
            with self._lock:
                msg = self.message
            # Clear line and write new status
            sys.stdout.write(f"\r\033[K{frame} {msg}")
            sys.stdout.flush()
            time.sleep(0.08)
        # Clear the spinner line when done
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)  # Don't wait forever


from rnow.cli.common import require_auth
from rnow.core.reward import REWARD_REGISTRY, clear_reward_registry, compute_total_reward
from rnow.core.tool import TOOL_REGISTRY, clear_tool_registry
from rnow.models import ProjectConfig, RewardArgs

DEFAULT_API_URL = "https://www.reinforcenow.ai"


class ModelCompleter:
    """
    Completer that handles tokenization and calls Next.js API.
    Requires authentication for billing.
    """

    def __init__(self, api_base: str, model: str, max_tokens: int = 2048, temperature: float = 1.0):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.auth_headers = get_auth_headers()
        self.client = httpx.AsyncClient(timeout=120.0)
        self.session_id: str | None = None  # Cached session ID for reuse
        self.total_latency_ms = 0
        self.request_count = 0

        # Initialize tokenizer and renderer
        from tinker_cookbook import renderers
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.tokenizer = get_tokenizer(model)
        renderer_name = get_recommended_renderer_name(model)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)

    async def __call__(self, messages: list[dict], stop: list[str] | None = None) -> dict:
        """
        Tokenize messages, call Next.js API, decode response.
        """
        # Build model input using renderer
        model_input = self.renderer.build_generation_prompt(messages)
        tokens = model_input.to_ints()

        # Get stop sequences from renderer if not provided
        if stop is None:
            stop = self.renderer.get_stop_sequences()

        # Build request payload
        payload = {
            "model": self.model,
            "tokens": tokens,
            "stop": stop,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Include session_id if we have one cached
        if self.session_id:
            payload["session_id"] = self.session_id

        # Call Next.js API with tokens
        resp = await self.client.post(
            f"{self.api_base}/api/rnow/sample",
            json=payload,
            headers=self.auth_headers,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise Exception(f"API error: {data.get('detail', data.get('error'))}")

        # Cache the session_id for future requests
        if "session_id" in data and data["session_id"]:
            self.session_id = data["session_id"]

        # Track latency
        if "latency_ms" in data:
            self.total_latency_ms += data["latency_ms"]
            self.request_count += 1

        # Decode tokens back to text
        output_tokens = data.get("tokens", [])
        parsed_message, _success = self.renderer.parse_response(output_tokens)

        return {
            "content": parsed_message.get("content", ""),
            "latency_ms": data.get("latency_ms", 0),
        }

    async def close(self):
        await self.client.aclose()


async def flush_pending_charges(api_url: str) -> dict | None:
    """
    Flush any pending ROLLOUT charges at the end of the test.
    Returns the flush result or None if it failed.
    """
    try:
        headers = get_auth_headers()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_url.rstrip('/')}/api/billing/flush-rollout",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        click.echo(click.style(f"Warning: Failed to flush pending charges: {e}", fg="yellow"))
        return None


def _exec_file(path: Path, module_name: str) -> None:
    """Execute a Python file to populate registries."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _build_tools_block(tool_registry: dict[str, Callable]) -> str:
    """Build the tools description block from registered tool functions."""
    if not tool_registry:
        return ""

    tools_json = []
    for name, fn in tool_registry.items():
        schema = getattr(fn, "_schema", {"type": "object", "properties": {}})
        description = getattr(fn, "_description", "No description available.")
        tools_json.append(
            {
                "name": name,
                "description": description,
                "parameters": schema,
            }
        )

    tools_block = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(tools_json, indent=2)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<value>"}}}}
</tool_call>
"""

    return tools_block


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _format_message(msg: dict, max_len: int = 300) -> str:
    """Format a message for display."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    # Truncate long content
    if len(content) > max_len:
        content = content[:max_len] + "..."
    # Color based on role
    colors = {"system": "yellow", "user": "blue", "assistant": "green", "tool": "magenta"}
    color = colors.get(role, "white")
    return click.style(f"[{role}]", fg=color) + f" {content}"


async def _run_single_rollout(
    completer: ModelCompleter,
    sample: dict,
    reward_registry: dict[str, Callable],
    tool_registry: dict[str, Callable],
    max_turns: int,
    termination_policy: str,
    verbose: bool = False,
) -> dict:
    """Run a single rollout for an RL sample."""
    import inspect

    messages_templates = sample["messages"]
    reward_names = sample["rewards"]
    variables = sample.get("variables", {})
    metadata = sample.get("metadata", {})

    reward_fns = []
    for name in reward_names:
        if name not in reward_registry:
            raise ValueError(f"Reward function '{name}' not found in registry")
        reward_fns.append(reward_registry[name])

    ctx = {**metadata, **variables}
    messages = [
        {"role": msg["role"], "content": Template(msg["content"]).safe_substitute(ctx)}
        for msg in messages_templates
    ]

    if tool_registry:
        tools_block = _build_tools_block(tool_registry)
        system_found = False
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = tools_block + "\n\n" + msg["content"]
                system_found = True
                break
        if not system_found:
            messages.insert(0, {"role": "system", "content": tools_block})

    conversation = messages.copy()
    turn_count = 0
    total_tool_calls = 0

    # Show initial messages in verbose mode
    if verbose:
        click.echo("  --- Initial Messages ---")
        for msg in messages:
            click.echo(f"    {_format_message(msg)}")
        click.echo("  -------------------------")

    while turn_count < max_turns:
        turn_count += 1

        result = await completer(conversation, stop=None)
        response_content = result.get("content", "")

        conversation.append({"role": "assistant", "content": response_content})

        if verbose:
            click.echo(
                f"  [Turn {turn_count}] {_format_message({'role': 'assistant', 'content': response_content}, max_len=500)}"
            )

        tool_matches = TOOL_CALL_RE.findall(response_content)
        tool_call_count = len(tool_matches)
        total_tool_calls += tool_call_count

        for raw_call in tool_matches:
            if not tool_registry:
                break
            try:
                tool_data = json.loads(raw_call)
                tool_name = tool_data.get("name")
                args = tool_data.get("arguments", {})

                if tool_name not in tool_registry:
                    tool_response = f"<tool_error>Tool '{tool_name}' not found</tool_error>"
                    conversation.append({"role": "tool", "content": tool_response})
                    if verbose:
                        click.echo(
                            f"    {_format_message({'role': 'tool', 'content': tool_response})}"
                        )
                    continue

                tool_fn = tool_registry[tool_name]
                tool_result = (
                    await tool_fn(**args)
                    if inspect.iscoroutinefunction(tool_fn)
                    else tool_fn(**args)
                )

                tool_response = f"<tool_result>{json.dumps(tool_result)}</tool_result>"
                conversation.append({"role": "tool", "content": tool_response})

                if verbose:
                    click.echo(
                        f"    Tool {click.style(tool_name, fg=TEAL_RGB)}: {str(tool_result)[:200]}"
                    )

            except json.JSONDecodeError as e:
                tool_response = f"<tool_error>Invalid JSON: {str(e)}</tool_error>"
                conversation.append({"role": "tool", "content": tool_response})
                if verbose:
                    click.echo(f"    {_format_message({'role': 'tool', 'content': tool_response})}")
            except Exception as e:
                tool_response = f"<tool_error>{str(e)}</tool_error>"
                conversation.append({"role": "tool", "content": tool_response})
                if verbose:
                    click.echo(f"    {_format_message({'role': 'tool', 'content': tool_response})}")

        if termination_policy == "last_tool" and tool_call_count == 0:
            break

    # Show final conversation summary in verbose mode
    if verbose:
        click.echo(f"  --- Rollout Complete: {turn_count} turns, {total_tool_calls} tool calls ---")

    reward_args = RewardArgs(metadata=metadata, variables=variables)
    rewards = {}
    for fn, name in zip(reward_fns, reward_names, strict=False):
        value = await fn(reward_args, conversation)
        rewards[name] = value

    total_reward = compute_total_reward(rewards) if rewards else 0.0

    return {
        "total_reward": total_reward,
        "rewards": rewards,
        "turns": turn_count,
        "tools_used": total_tool_calls,
        "conversation": conversation,
    }


def _check_test_dependencies():
    """Check if optional test dependencies are installed."""
    try:
        import tinker_cookbook  # noqa: F401
    except ImportError:
        click.echo()
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + "The 'rnow test' command requires additional dependencies."
        )
        pip_cmd = "pip install 'rnow[test]'"
        click.echo(f"Install them with: {click.style(pip_cmd, fg=TEAL_RGB)}")
        click.echo()
        raise SystemExit(1)


@click.command(name="test")
@click.option(
    "--dir",
    "-d",
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    help="Project directory containing config.yml, rewards.py, env.py, train.jsonl",
)
@click.option(
    "--num-rollouts",
    "-n",
    default=3,
    show_default=True,
    help="Number of rollouts to run",
)
@click.option(
    "--multi-turn/--single-turn",
    default=True,
    show_default=True,
    help="Allow multi-turn rollouts or force single-turn",
)
@click.option(
    "--with-tools/--no-tools",
    default=True,
    show_default=True,
    help="Enable or disable tool use during rollout",
)
@click.option(
    "--model",
    default=None,
    help="Override model name for sampling (otherwise uses config.model.path)",
)
@click.option(
    "--api-url",
    envvar="RNOW_API_URL",
    default=None,
    help="Base URL of the Next.js backend (default: https://www.reinforcenow.ai)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed output for each rollout turn",
)
@click.option(
    "--truncate",
    "-t",
    default=None,
    type=int,
    help="Truncate message content to N characters (default: no truncation)",
)
@click.pass_context
def test(ctx, project_dir, num_rollouts, multi_turn, with_tools, model, api_url, verbose, truncate):
    """Test RL rollouts locally before submitting.

    This command runs local RL rollouts by calling the Next.js API
    for model sampling.

    Only works with RL projects (dataset_type: rl).
    """
    global _shutdown_requested
    _shutdown_requested = False

    def handle_sigint(signum, frame):
        global _shutdown_requested
        if _shutdown_requested:
            # Second Ctrl+C, force exit
            sys.exit(1)
        _shutdown_requested = True
        click.echo("\n" + click.style("Interrupted. Shutting down gracefully...", fg="yellow"))

    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, handle_sigint)

    require_auth()
    _check_test_dependencies()
    try:
        asyncio.run(
            _test_async(
                project_dir=project_dir,
                num_rollouts=num_rollouts,
                multi_turn=multi_turn,
                with_tools=with_tools,
                model_override=model,
                api_url=api_url
                or ctx.obj.get("api_url", "").replace("/api", "")
                or DEFAULT_API_URL,
                verbose=verbose,
                truncate=truncate,
            )
        )
    except KeyboardInterrupt:
        click.echo(click.style("Aborted.", fg="yellow"))
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


async def _test_async(
    project_dir: Path,
    num_rollouts: int,
    multi_turn: bool,
    with_tools: bool,
    model_override: str | None,
    api_url: str,
    verbose: bool,
    truncate: int | None,
):
    project_dir = Path(project_dir)

    config_path = project_dir / "config.yml"
    if not config_path.exists():
        config_path = project_dir / "config.json"

    if not config_path.exists():
        raise click.ClickException("No config.yml or config.json found in project directory")

    if config_path.suffix == ".yml":
        config_data = yaml.safe_load(config_path.read_text())
    else:
        config_data = json.loads(config_path.read_text())

    config = ProjectConfig(**config_data)

    if config.dataset_type.value != "rl":
        raise click.ClickException(
            f"rnow test only supports RL projects (dataset_type: rl). "
            f"Found: {config.dataset_type.value}"
        )

    rewards_path = project_dir / "rewards.py"
    env_path = project_dir / "env.py"
    train_path = project_dir / "train.jsonl"

    if not rewards_path.exists():
        raise click.ClickException("rewards.py not found in project directory")
    if not train_path.exists():
        raise click.ClickException("train.jsonl not found in project directory")

    # Validate max_tokens vs prompt size
    from rnow.cli.commands import get_max_prompt_tokens, validate_max_tokens_for_context
    from rnow.models import MAX_CONTEXT_WINDOW

    if config.rollout:
        max_prompt_tokens = get_max_prompt_tokens(train_path)
        if max_prompt_tokens > 0:
            context_error, recommended = validate_max_tokens_for_context(
                config.rollout.max_tokens, max_prompt_tokens
            )
            if context_error:
                click.echo()
                click.echo(click.style("✗ Context window exceeded", fg="red", bold=True))
                click.echo()
                click.echo(
                    f"  Your longest prompt in train.jsonl is ~{max_prompt_tokens:,} tokens."
                )
                click.echo(f"  With max_tokens={config.rollout.max_tokens:,}, the total exceeds")
                click.echo(f"  the {MAX_CONTEXT_WINDOW:,} token context window.")
                click.echo()
                click.echo(
                    click.style("  Fix:", bold=True)
                    + f" Set rollout.max_tokens to {recommended:,} or less"
                )
                click.echo()
                raise click.ClickException("max_tokens + prompt length exceeds context window")

    clear_reward_registry()
    clear_tool_registry()

    _exec_file(rewards_path, "rewards")

    if with_tools and env_path.exists():
        _exec_file(env_path, "env")

    samples = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]

    if not samples:
        raise click.ClickException("train.jsonl is empty")

    model_name = model_override or config.model.path
    max_tokens = config.rollout.max_tokens if config.rollout else 2048
    max_turns_config = config.rollout.max_turns if config.rollout else 1
    termination_policy = config.rollout.termination_policy if config.rollout else "last_tool"

    max_turns = 1 if not multi_turn else max_turns_config

    # Check for gpt-oss with tools - not supported in rnow test
    is_gpt_oss = "gpt-oss" in model_name.lower() or "gptoss" in model_name.lower()
    has_tools = with_tools and (env_path.exists() or (config.rollout and config.rollout.mcp_url))

    if is_gpt_oss and has_tools:
        click.echo(
            click.style("Warning: ", fg="yellow")
            + "Tool calling with gpt-oss models is not supported in 'rnow test'. Running without tools."
        )
        with_tools = False

    rewards = []
    tool_registry_to_use = TOOL_REGISTRY if with_tools else {}

    # Display model info with reasoning mode (same format as rnow run)
    thinking_display = get_thinking_mode_display(config)
    click.echo(f"Model: {model_name} ({click.style(thinking_display, fg=TEAL_RGB)})")
    click.echo()

    try:
        # Create one completer per concurrent rollout to avoid session conflicts
        completers = [
            ModelCompleter(
                api_base=api_url,
                model=model_name,
                max_tokens=max_tokens,
            )
            for _ in range(num_rollouts)
        ]

        # Select samples for each rollout upfront
        selected_samples = [random.choice(samples) for _ in range(num_rollouts)]

        # Start spinner for concurrent rollouts
        spinner = Spinner(f"Running {num_rollouts} rollouts concurrently...")
        spinner.start()

        async def run_rollout_with_index(idx: int) -> tuple[int, dict | Exception]:
            """Run a single rollout and return (index, result or exception)."""
            if _shutdown_requested:
                return (idx, asyncio.CancelledError("Shutdown requested"))
            try:
                result = await _run_single_rollout(
                    completer=completers[idx],
                    sample=selected_samples[idx],
                    reward_registry=REWARD_REGISTRY,
                    tool_registry=tool_registry_to_use,
                    max_turns=max_turns,
                    termination_policy=termination_policy,
                    verbose=False,
                )
                return (idx, result)
            except asyncio.CancelledError:
                return (idx, asyncio.CancelledError("Cancelled"))
            except Exception as e:
                return (idx, e)

        # Run all rollouts concurrently
        start_time = time.time()
        tasks = [asyncio.create_task(run_rollout_with_index(i)) for i in range(num_rollouts)]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # Cancel all tasks if we get interrupted
            for task in tasks:
                task.cancel()
            results = []

        total_time = time.time() - start_time
        spinner.stop()

        # Check if shutdown was requested
        if _shutdown_requested:
            # Close completers and exit early
            for c in completers:
                await c.close()
            return

        # Display results in order
        for idx, result in sorted(results, key=lambda x: x[0]):
            click.echo(f"Rollout {idx+1}/{num_rollouts}")

            if isinstance(result, Exception):
                if isinstance(result, httpx.HTTPStatusError):
                    click.echo(
                        click.style(f"  ✗ HTTP Error: {result.response.status_code}", fg="red")
                    )
                else:
                    click.echo(click.style(f"  ✗ {result}", fg="red"))
                click.echo()
                continue

            total_reward = result["total_reward"]
            rewards.append(total_reward)

            # Get conversation
            conversation = result["conversation"]

            # Show all messages with red tags
            for msg in conversation:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate if flag is set
                if truncate and len(content) > truncate:
                    content = content[:truncate] + "..."
                tag = click.style(f"[{role}]", fg="red")
                click.echo(f"  {tag} {content}")
            reward_str = ", ".join(f"{k}={v:.3f}" for k, v in result["rewards"].items())
            click.echo(
                f"  {click.style('reward', fg=TEAL_RGB)}={total_reward:.3f} "
                f"| turns={result['turns']} "
                f"| tools_used={result['tools_used']} "
                f"| [{reward_str}]"
            )
            click.echo()

        # Close all completers
        for c in completers:
            await c.close()

    except Exception:
        raise

    # Flush any pending billing charges
    flush_result = await flush_pending_charges(api_url)

    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        click.echo()
        click.echo(f"Mean reward: {click.style(f'{mean_reward:.3f}', fg=TEAL_RGB)}")
        click.echo(f"Latency: {click.style(f'{total_time:.1f}s', fg=TEAL_RGB)}")
    else:
        click.echo(click.style("\nNo successful rollouts completed.", fg="yellow"))

    # Show billing summary if charges were flushed
    if flush_result and flush_result.get("flushed"):
        amount_cents = flush_result.get("amountCents", 0)
        total_tokens = flush_result.get("totalTokens", 0)
        click.echo(
            f"Billing: {click.style(f'${amount_cents/100:.2f}', fg=TEAL_RGB)} ({total_tokens:,} tokens)"
        )
