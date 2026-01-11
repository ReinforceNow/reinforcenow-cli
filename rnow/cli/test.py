# rnow/cli/test.py
"""
Test command for running RL rollouts via API.

Uses the /api/rnow/rollout endpoint which runs rollouts in Modal sandbox.

Modes:
- Default: Uses tinker models (requires auth)
- --smoke-test: Uses OpenAI gpt-5-nano (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

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
from rnow.models import ProjectConfig

DEFAULT_API_URL = "https://www.reinforcenow.ai"


class RolloutClient:
    """
    Client for running rollouts via the /api/rnow/rollout endpoint.

    Uses the same tools.py rollout logic as production training,
    ensuring single source of truth.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        max_turns: int = 1,
        termination_policy: str = "last_tool",
        debug: bool = False,
        smoke_test: bool = False,
        openai_api_key: str | None = None,
        mcp_url: str | list[str] | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_turns = max_turns
        self.termination_policy = termination_policy
        self.debug = debug
        self.smoke_test = smoke_test
        self.openai_api_key = openai_api_key
        self.mcp_url = mcp_url
        self.auth_headers = get_auth_headers()
        self.client = httpx.AsyncClient(timeout=3600.0)  # 60 min for batch rollouts
        self.total_charged_dollars = 0.0

    async def run_batch_rollouts(
        self,
        samples: list[dict],
        tools_py_code: str | None = None,
        rewards_py_code: str | None = None,
    ) -> list[dict]:
        """
        Run multiple rollouts in parallel via a single API call.

        Uses one sampler (policy) for all rollouts, running them
        concurrently with asyncio.gather on the server side.
        """
        payload = {
            "samples": samples,  # Batch mode
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "termination_policy": self.termination_policy,
            "tools_py_code": tools_py_code,
            "rewards_py_code": rewards_py_code,
            "debug": self.debug,
        }

        # Add MCP URL if configured
        if self.mcp_url:
            payload["mcp_url"] = self.mcp_url

        # Smoke test mode: use OpenAI instead of tinker
        if self.smoke_test:
            payload["smoke_test"] = True
            payload["openai_api_key"] = self.openai_api_key

        resp = await self.client.post(
            f"{self.api_base}/api/rnow/rollout",
            json=payload,
            headers=self.auth_headers,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise Exception(f"API error: {data.get('detail', data.get('error'))}")

        # Track billing
        if "billing" in data:
            billing = data["billing"]
            tokens = billing.get("prompt_tokens", 0) + billing.get("completion_tokens", 0)
            self.total_charged_dollars += tokens * 0.000001

        # Return results array
        return data.get("results", [data])  # Fallback to single result for backwards compat

    async def close(self):
        await self.client.aclose()


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
    client: RolloutClient,
    sample: dict,
    tools_py_code: str | None,
    rewards_py_code: str | None,
    verbose: bool = False,
) -> dict:
    """Run a single rollout via the API."""
    result = await client.run_rollout(
        sample=sample,
        tools_py_code=tools_py_code,
        rewards_py_code=rewards_py_code,
    )

    # Show conversation in verbose mode
    if verbose:
        click.echo("  --- Conversation ---")
        for msg in result.get("conversation", []):
            click.echo(f"    {_format_message(msg)}")
        click.echo("  ---------------------")

    return result


@click.command(name="test")
@click.option(
    "--dir",
    "-d",
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    help="Project directory containing config.yml, rewards.py, tools.py, train.jsonl",
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
@click.option(
    "--debug",
    is_flag=True,
    help="Use debug trainer image from Docker Hub (for testing trainer changes)",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Save rollout results as JSON files in this directory",
)
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Use OpenAI gpt-5-nano instead of tinker (requires OPENAI_API_KEY env var)",
)
@click.pass_context
def test(
    ctx,
    project_dir,
    num_rollouts,
    multi_turn,
    with_tools,
    model,
    api_url,
    verbose,
    truncate,
    debug,
    output_dir,
    smoke_test,
):
    """Test RL rollouts before submitting.

    Runs rollouts via the /api/rnow/rollout endpoint in Modal sandbox.

    Use --smoke-test to use OpenAI gpt-5-nano instead of tinker models
    (requires OPENAI_API_KEY environment variable).

    Only works with RL projects (dataset_type: rl).
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Check for OpenAI API key in smoke test mode
    openai_api_key = None
    if smoke_test:
        # Get API key from environment variable
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise click.ClickException(
                "OPENAI_API_KEY environment variable is required for smoke test mode.\n"
                "Set it with: export OPENAI_API_KEY=sk-..."
            )
    else:
        require_auth()

    async def run_with_cancellation():
        """Run test with proper cancellation support."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()

        def handle_sigint():
            global _shutdown_requested
            if _shutdown_requested:
                # Second Ctrl+C, force exit
                sys.exit(1)
            _shutdown_requested = True
            click.echo("\n" + click.style("Interrupted. Cancelling...", fg="yellow"))
            task.cancel()

        # Add signal handler to the event loop
        loop.add_signal_handler(signal.SIGINT, handle_sigint)

        try:
            await _test_async(
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
                debug=debug,
                output_dir=output_dir,
                smoke_test=smoke_test,
                openai_api_key=openai_api_key,
            )
        except asyncio.CancelledError:
            click.echo(click.style("Aborted.", fg="yellow"))
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    try:
        asyncio.run(run_with_cancellation())
    except KeyboardInterrupt:
        click.echo(click.style("Aborted.", fg="yellow"))


async def _test_async(
    project_dir: Path,
    num_rollouts: int,
    multi_turn: bool,
    with_tools: bool,
    model_override: str | None,
    api_url: str,
    verbose: bool,
    truncate: int | None,
    debug: bool = False,
    output_dir: Path | None = None,
    smoke_test: bool = False,
    openai_api_key: str | None = None,
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
    tools_path = project_dir / "tools.py"
    train_path = project_dir / "train.jsonl"

    if not rewards_path.exists():
        raise click.ClickException("rewards.py not found in project directory")
    if not train_path.exists():
        raise click.ClickException("train.jsonl not found in project directory")

    # Read user code files to send to the API
    rewards_py_code = rewards_path.read_text()
    tools_py_code = tools_path.read_text() if with_tools and tools_path.exists() else None

    # Load samples
    samples = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]

    if not samples:
        raise click.ClickException("train.jsonl is empty")

    # For smoke test, always use gpt-5-nano
    model_name = "gpt-5-nano" if smoke_test else model_override or config.model.path

    max_tokens = config.rollout.max_tokens if config.rollout else 2048
    max_turns_config = config.rollout.max_turns if config.rollout else 1
    termination_policy = config.rollout.termination_policy if config.rollout else "last_tool"
    mcp_url = config.rollout.mcp_url if config.rollout else None

    max_turns = 1 if not multi_turn else max_turns_config

    # Display mode and model info
    if smoke_test:
        click.echo(f"Mode: {click.style('SMOKE TEST', fg=TEAL_RGB)} (OpenAI gpt-5-nano)")
    else:
        thinking_display = get_thinking_mode_display(config)
        click.echo(f"Model: {model_name} ({click.style(thinking_display, fg=TEAL_RGB)})")

    # Display MCP info if configured
    if mcp_url:
        if isinstance(mcp_url, list):
            click.echo(f"MCP: {len(mcp_url)} server(s)")
        elif mcp_url.startswith("docker://"):
            click.echo(f"MCP: {click.style(mcp_url, fg='cyan')} (Modal sandbox)")
        else:
            click.echo(f"MCP: {mcp_url}")
    click.echo()

    rewards = []

    try:
        # Create one RolloutClient for all rollouts
        client = RolloutClient(
            api_base=api_url,
            model=model_name,
            max_tokens=max_tokens,
            temperature=1.0,
            max_turns=max_turns,
            termination_policy=termination_policy,
            debug=debug,
            smoke_test=smoke_test,
            openai_api_key=openai_api_key,
            mcp_url=mcp_url,
        )

        # Select samples for batch rollout
        selected_samples = [random.choice(samples) for _ in range(num_rollouts)]

        # Start spinner for batch rollout
        spinner = Spinner(f"Running {num_rollouts} rollouts...")
        spinner.start()

        start_time = time.time()

        try:
            # Single API call - server runs all rollouts in parallel with one sampler
            batch_results = await client.run_batch_rollouts(
                samples=selected_samples,
                tools_py_code=tools_py_code,
                rewards_py_code=rewards_py_code,
            )
        except asyncio.CancelledError:
            batch_results = []
        except Exception as e:
            spinner.stop()
            raise e

        total_time = time.time() - start_time
        spinner.stop()

        # Check if shutdown was requested
        if _shutdown_requested:
            await client.close()
            return

        # Save results to files if output_dir is specified
        if output_dir and batch_results:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save individual rollout results
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    continue
                filename = output_dir / f"rollout_{timestamp}_{idx+1}.json"
                filename.write_text(json.dumps(result, indent=2))

            # Save summary with all results
            summary = {
                "model": model_name,
                "num_rollouts": num_rollouts,
                "max_turns": max_turns,
                "timestamp": timestamp,
                "total_time_seconds": total_time,
                "results": [r for r in batch_results if not isinstance(r, Exception)],
            }
            summary_file = output_dir / f"summary_{timestamp}.json"
            summary_file.write_text(json.dumps(summary, indent=2))
            click.echo(f"Results saved to {click.style(str(output_dir), fg=TEAL_RGB)}")
            click.echo()

        # Display results in order
        for idx, result in enumerate(batch_results):
            click.echo(f"Rollout {idx+1}/{num_rollouts}")

            if isinstance(result, Exception):
                if isinstance(result, httpx.HTTPStatusError):
                    click.echo(
                        click.style(f"  ✗ HTTP Error: {result.response.status_code}", fg="red")
                    )
                    try:
                        error_detail = result.response.json()
                        click.echo(
                            f"    {error_detail.get('detail', error_detail.get('error', ''))}"
                        )
                    except Exception:
                        pass
                else:
                    click.echo(click.style(f"  ✗ {result}", fg="red"))
                click.echo()
                continue

            total_reward = result.get("total_reward", 0.0)
            rewards.append(total_reward)

            # Get conversation
            conversation = result.get("conversation", [])

            # Show all messages with red tags
            for msg in conversation:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate if flag is set
                if truncate and len(content) > truncate:
                    content = content[:truncate] + "..."
                tag = click.style(f"[{role}]", fg="red")
                click.echo(f"  {tag} {content}")

            reward_breakdown = result.get("rewards", {})
            reward_str = ", ".join(f"{k}={v:.3f}" for k, v in reward_breakdown.items())
            turns = result.get("turns", 0)
            click.echo(
                f"  {click.style('reward', fg=TEAL_RGB)}={total_reward:.3f} "
                f"| turns={turns} "
                f"| [{reward_str}]"
            )
            click.echo()

        # Get total billing
        total_charged = client.total_charged_dollars

        # Close client
        await client.close()

    except Exception:
        raise

    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        click.echo()
        click.echo(f"Mean reward: {click.style(f'{mean_reward:.3f}', fg=TEAL_RGB)}")
        click.echo(f"Latency: {click.style(f'{total_time:.1f}s', fg=TEAL_RGB)}")
        if total_charged > 0:
            click.echo(f"Cost: {click.style(f'${total_charged:.4f}', fg=TEAL_RGB)}")
    else:
        click.echo(click.style("\nNo successful rollouts completed.", fg="yellow"))
