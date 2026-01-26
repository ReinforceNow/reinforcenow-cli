# rnow/cli/test.py
"""
Test command for running RL rollouts via API.

Uses the /api/rnow/rollout endpoint which runs rollouts on Cloud Run.
Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import signal
import sys
import time
import uuid
from pathlib import Path

import click
import httpx
import yaml
from rich.console import Console
from rich.live import Live
from rich.text import Text

# Global flag for graceful shutdown
_shutdown_requested = False

from rnow.cli.auth import get_auth_headers

# ReinforceNow teal
TEAL = "#14B8A6"
TEAL_RGB = (20, 184, 166)  # For click.style()

console = Console()


from rnow.models import SUPPORTED_MODELS_SET, ProjectConfig

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

DEFAULT_API_URL = "https://www.reinforcenow.ai"


def is_gpu_model(model: str) -> bool:
    """Check if model requires GPU infrastructure vs OpenAI API.

    GPU models: Models in SUPPORTED_MODELS_SET (Qwen, Llama, DeepSeek, etc.)
    OpenAI models: gpt-5.2, gpt-5-mini, gpt-5-nano, gpt-5-pro (and snapshots)
    """
    # Model IDs (UUIDs) are finetuned models that require GPU
    if _looks_like_model_id(model):
        return True
    return model in SUPPORTED_MODELS_SET


def _looks_like_model_id(model: str) -> bool:
    """Check if model looks like a ReinforceNow model ID (UUID format)."""
    import re

    # UUID format: 8-4-4-4-12 hex characters
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(uuid_pattern, model.lower()))


class RolloutClient:
    """
    Client for running rollouts via the /api/rnow/rollout endpoint.

    Uses SSE streaming: POST starts job and returns tunnel URL, CLI connects to SSE.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        max_context_window: int = 32768,
        temperature: float = 1.0,
        max_turns: int = 1,
        termination_policy: str = "last_tool",
        max_tool_response: int | None = None,
        debug: bool = False,
        smoke_test: bool = False,
        openai_api_key: str | None = None,
        mcp_url: str | list[str] | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_context_window = max_context_window
        self.temperature = temperature
        self.max_turns = max_turns
        self.termination_policy = termination_policy
        self.max_tool_response = max_tool_response
        self.debug = debug
        self.smoke_test = smoke_test
        self.openai_api_key = openai_api_key
        self.mcp_url = mcp_url
        self.auth_headers = get_auth_headers()
        self.client = httpx.AsyncClient(timeout=120.0)
        self.total_charged_dollars = 0.0

    async def run_batch_rollouts(
        self,
        samples: list[dict],
        tools_py_code: str | None = None,
        rewards_py_code: str | None = None,
        requirements_txt: str | None = None,
        dockerfiles: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        timeout_minutes: int = 30,
        start_time: float | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Run rollouts with SSE streaming via the API streaming endpoint.
        Returns (rollout_id, results).

        Shows live status table with spinner, prints conversations as they stream.
        """
        # Use provided start_time (from command start) or fall back to now
        if start_time is None:
            start_time = time.time()

        # Track state
        num_samples = len(samples)
        results_by_index: dict[int, dict] = {}
        status: dict[int, str] = {
            i: "queued" for i in range(num_samples)
        }  # queued, running, done, error
        spinner_frame = 0
        current_phase = "connecting"  # connecting, streaming

        # Create rollouts directory and find next run number
        rollouts_dir = Path("rollouts")
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Find the next available run number (rollout_1, rollout_2, etc.)
        existing_runs = [d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("rollout_")]
        existing_nums = []
        for d in existing_runs:
            try:
                num = int(d.name.split("_")[1])
                existing_nums.append(num)
            except (IndexError, ValueError):
                pass
        run_num = max(existing_nums, default=0) + 1
        run_folder = rollouts_dir / f"rollout_{run_num}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Track files and conversations (for streaming writes)
        rollout_files: dict[int, Path] = {}
        rollout_ids: dict[int, str] = {}
        rollout_metadata: dict[int, dict] = {}  # Store metadata separately
        conversations: dict[int, list] = {}

        # Field order: metadata first, conversation last
        FIELD_ORDER = [
            "id", "completed", "success",  # Status
            "total_reward", "rewards", "errors",  # Rewards
            "turns", "truncated",  # Rollout info
            "totalTokens", "promptTokens", "completion",  # Token usage
            "metadata",  # Entry metadata
        ]

        def write_rollout_file(idx: int, data: dict) -> None:
            """Write rollout state with metadata at top, conversation at end."""
            if idx not in rollout_files:
                return
            ordered = {k: data[k] for k in FIELD_ORDER if k in data}
            ordered.update({k: v for k, v in data.items() if k not in ordered and k != "conversation"})
            if "conversation" in data:
                ordered["conversation"] = data["conversation"]
            rollout_files[idx].write_text(json.dumps(ordered, indent=2))

        def render_status() -> Text:
            nonlocal spinner_frame
            spinner_frame += 1
            spinner = SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)]
            elapsed = int(time.time() - start_time)

            text = Text()

            # Show rollout statuses
            for i in range(num_samples):
                s = status[i]
                if s == "queued":
                    text.append(f"Rollout {i + 1}: ", style=f"bold {TEAL}")
                    text.append("queued\n", style="dim")
                elif s == "running":
                    rid = rollout_ids.get(i, "")
                    text.append(f"Rollout {i + 1}: ", style=f"bold {TEAL}")
                    text.append("streaming…", style="white")
                    if rid:
                        text.append(f"  → rollouts/rollout_{run_num}/{rid}.json", style="dim")
                    text.append("\n")
                elif s == "done":
                    result = results_by_index.get(i, {})
                    reward = result.get("total_reward")
                    breakdown = result.get("rewards", {})
                    rid = rollout_ids.get(i, "")
                    reward_errors = result.get("errors", [])

                    # Check if any reward returned "error" or "timeout"
                    has_reward_error = any(
                        v in ("error", "timeout") for v in breakdown.values()
                    ) if breakdown else False

                    text.append(f"Rollout {i + 1}: ", style=f"bold {TEAL}")

                    # Show red ✗ if any reward errored, green ✓ otherwise
                    if has_reward_error:
                        text.append("✗", style="red")
                        # When rewards error, effective reward is 0
                        display_reward = 0.0
                    else:
                        text.append("✓", style="green")
                        display_reward = reward

                    # Show reward if available (RL mode)
                    if reward is not None:
                        reward_style = "red" if has_reward_error else "green"
                        text.append(f" reward={display_reward:.3f}", style=reward_style)
                        if breakdown:
                            # Handle both float values and string values (like "timeout" or "error")
                            bd = ", ".join(
                                f"{k}={v:.3f}" if isinstance(v, int | float) else f"{k}={v}"
                                for k, v in breakdown.items()
                            )
                            text.append(f"  [{bd}]", style=reward_style)
                    if rid:
                        text.append(f"  → rollouts/rollout_{run_num}/{rid}.json", style="dim")

                    text.append("\n")

                    # Show reward errors (e.g., LLM judge failures)
                    if reward_errors:
                        for err in reward_errors:
                            text.append(f"    ⚠ {err}\n", style="yellow")

                    # Show tool errors if any (helps debug API key issues, etc.)
                    tool_errors = result.get("tool_errors", [])
                    if tool_errors:
                        for err in tool_errors:
                            text.append(f"    ⚠ {err}\n", style="yellow")
                elif s == "error":
                    result = results_by_index.get(i, {})
                    error = result.get("error", "Unknown")
                    text.append(f"Rollout {i + 1}: ", style=f"bold {TEAL}")
                    text.append(f"✗ {error}\n", style="red")

            pending = sum(1 for s in status.values() if s in ("queued", "running"))
            if pending > 0:
                if current_phase == "connecting":
                    text.append(
                        f"\n{spinner} Connecting… ({elapsed}s)\n\n",
                        style="white",
                    )
                else:
                    text.append(
                        f"\n{spinner} Running… {num_samples - pending}/{num_samples} complete ({elapsed}s)\n\n",
                        style="white",
                    )
            else:
                text.append(f"\n✓ Complete ({elapsed}s)\n\n", style="green")

            return text

        done_streaming = False
        rollout_id = ""

        # Start the Live display immediately
        with Live(render_status(), console=console, refresh_per_second=10) as live:
            # Background task to update spinner and timer
            async def update_display():
                while not done_streaming:
                    live.update(render_status())
                    await asyncio.sleep(0.1)

            update_task = asyncio.create_task(update_display())

            try:
                # Build payload for Next.js API
                payload = {
                    "samples": samples,
                    "model": self.model,
                    "max_context_window": self.max_context_window,
                    "temperature": self.temperature,
                    "max_turns": self.max_turns,
                    "termination_policy": self.termination_policy,
                    "rewards_py_code": rewards_py_code,
                    "tools_py_code": tools_py_code,
                    "requirements_txt": requirements_txt,
                    "secrets": secrets,
                    "dockerfiles": dockerfiles,
                }

                if self.mcp_url:
                    payload["mcp_url"] = self.mcp_url

                if self.max_tool_response is not None:
                    payload["max_tool_response"] = self.max_tool_response

                # Get streaming URL and payload from Next.js API
                resp = await self.client.post(
                    f"{self.api_base}/api/rnow/rollout",
                    json=payload,
                    headers=self.auth_headers,
                )
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise Exception(f"API error: {data.get('error')}")

                rollout_id = data["rollout_id"]
                streaming_url = data["streaming_url"]
                streaming_payload = data["payload"]

                # Now connect to the streaming endpoint
                async with (
                    httpx.AsyncClient(timeout=None) as sse_client,
                    sse_client.stream(
                        "POST",
                        streaming_url,
                        json=streaming_payload,
                        headers={"Accept": "text/event-stream"},
                    ) as response,
                ):
                    response.raise_for_status()
                    current_phase = "streaming"
                    buffer = ""

                    async for chunk in response.aiter_text():
                        if _shutdown_requested:
                            raise asyncio.CancelledError()

                        buffer += chunk

                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            event_data = self._parse_sse_event(event_str)

                            if event_data is None:
                                continue

                            event_type = event_data.get("event")
                            data_json = event_data.get("data")

                            if event_type == "rollout_start":
                                idx = data_json.get("index", 0)
                                status[idx] = "running"

                                # Generate filename: sample_1.json, sample_2.json, etc.
                                sample_num = idx + 1
                                result_id = f"sample_{sample_num}"
                                rollout_ids[idx] = result_id
                                rollout_files[idx] = run_folder / f"{result_id}.json"

                                # Initialize conversation with initial messages
                                initial_messages = data_json.get("messages", [])
                                conversations[idx] = list(initial_messages)

                                # Store metadata from the sample if available
                                rollout_metadata[idx] = data_json.get("metadata", {})

                                # Write initial state
                                write_rollout_file(
                                    idx,
                                    {
                                        "id": result_id,
                                        "completed": False,
                                        "metadata": rollout_metadata[idx],
                                        "conversation": conversations[idx],
                                    },
                                )

                            elif event_type == "message":
                                # Append message to conversation and update file
                                idx = data_json.get("index", 0)
                                msg = data_json.get("message", {})
                                if idx in conversations and msg:
                                    conversations[idx].append(msg)
                                    write_rollout_file(
                                        idx,
                                        {
                                            "id": rollout_ids.get(idx, ""),
                                            "completed": False,
                                            "metadata": rollout_metadata.get(idx, {}),
                                            "conversation": conversations[idx],
                                        },
                                    )

                            elif event_type == "result":
                                idx = data_json.get("index", 0)

                                # Use existing ID or generate new one
                                result_id = rollout_ids.get(idx)
                                if not result_id:
                                    sample_num = idx + 1
                                    result_id = f"sample_{sample_num}"
                                    rollout_ids[idx] = result_id
                                    rollout_files[idx] = run_folder / f"{result_id}.json"

                                data_json["id"] = result_id
                                # Add completed flag: true for success, "error" for failure
                                data_json["completed"] = (
                                    True if data_json.get("success") else "error"
                                )
                                results_by_index[idx] = data_json

                                if data_json.get("success"):
                                    status[idx] = "done"
                                else:
                                    status[idx] = "error"

                                # Write final result with rewards
                                write_rollout_file(idx, data_json)

                            elif event_type == "complete":
                                billing = data_json.get("billing", {})
                                tokens = billing.get("prompt_tokens", 0) + billing.get(
                                    "completion_tokens", 0
                                )
                                self.total_charged_dollars += tokens * 0.000001

            except httpx.RemoteProtocolError as e:
                console.print(f"[yellow]Connection closed: {e}[/yellow]")

            finally:
                done_streaming = True
                update_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await update_task
                # Final update
                live.update(render_status())

        return rollout_id, list(results_by_index.values())

    def _parse_sse_event(self, event_str: str) -> dict | None:
        """Parse an SSE event string into event type and data."""
        event_type = None
        data_str = None

        for line in event_str.strip().split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()

        if data_str:
            try:
                data_json = json.loads(data_str)
                return {"event": event_type, "data": data_json}
            except json.JSONDecodeError:
                pass

        return None

    async def close(self):
        await self.client.aclose()


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
    default=1,
    show_default=True,
    help="Number of rollouts to run",
)
@click.option(
    "--model",
    default=None,
    help="Model for sampling. Use gpt-5 models (gpt-5-nano, gpt-5-mini, gpt-5.2, gpt-5-pro) "
    "or a finetuned model ID.",
)
@click.option(
    "--max-context-window",
    default=None,
    type=int,
    help="Override max context window (otherwise uses config.rollout.max_context_window)",
)
@click.option(
    "--api-url",
    envvar="RNOW_API_URL",
    default=None,
    hidden=True,
    help="Base URL of the Next.js backend",
)
@click.option(
    "--debug",
    is_flag=True,
    hidden=True,
    help="Use debug trainer image",
)
@click.option(
    "--entry",
    "-e",
    "entries",
    default=None,
    help="Entry indices from train.jsonl (0-indexed). Examples: -e 5, -e 0,2,5, -e 0 -e 2 -e 5",
    multiple=True,
)
@click.pass_context
def test(
    ctx,
    project_dir,
    num_rollouts,
    model,
    max_context_window,
    api_url,
    debug,
    entries,
):
    """Test RL rollouts before submitting.

    Runs rollouts via the ReinforceNow API. Requires OPENAI_API_KEY.

    Works with RL and SFT projects. For SFT/distillation, rewards.py is optional.
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Start timing immediately
    start_time = time.time()

    # Load .env file into os.environ (supports both .env and export)
    env_file = project_dir / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)  # Don't override existing env vars

    resolved_api_url = api_url or ctx.obj.get("api_url", "").replace("/api", "") or DEFAULT_API_URL

    # Get OpenAI API key (may not be needed for GPU models)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    async def run_with_cancellation():
        """Run test with proper cancellation support."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()

        def handle_sigint():
            global _shutdown_requested
            if _shutdown_requested:
                sys.exit(1)
            _shutdown_requested = True
            click.echo("\n" + click.style("Interrupted. Cancelling...", fg="yellow"))
            task.cancel()

        loop.add_signal_handler(signal.SIGINT, handle_sigint)

        try:
            await _test_async(
                project_dir=project_dir,
                num_rollouts=num_rollouts,
                model_override=model,
                max_context_window_override=max_context_window,
                api_url=resolved_api_url,
                debug=debug,
                openai_api_key=openai_api_key,
                entries=entries,
                start_time=start_time,
            )
        except asyncio.CancelledError:
            click.echo(click.style("Aborted.", fg="yellow"))
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    try:
        asyncio.run(run_with_cancellation())
    except KeyboardInterrupt:
        click.echo(click.style("Aborted.", fg="yellow"))


async def _fetch_rollout_results(
    rollout_id: str,
    api_url: str,
    store: bool = False,
    truncate: int | None = None,
    output_dir: Path | None = None,
):
    """Fetch results for an existing rollout ID."""
    click.echo(f"Fetching results for rollout: {click.style(rollout_id, fg=TEAL_RGB)}")

    client = httpx.AsyncClient(timeout=30.0)
    auth_headers = get_auth_headers()

    try:
        resp = await client.get(
            f"{api_url}/api/rnow/rollout",
            params={"id": rollout_id},
            headers=auth_headers,
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        await client.aclose()

    status = data.get("status")
    if status == "pending":
        click.echo(click.style("Rollout still running...", fg="yellow"))
        click.echo(f"Poll again with: rnow test --id {rollout_id}")
        return

    if status == "failed":
        raise click.ClickException(f"Rollout failed: {data.get('error', 'Unknown')}")

    # Store rollout ID if requested
    if store:
        _store_rollout_id(rollout_id, data)

    # Display results
    results = data.get("results", [])
    failed_count = _display_results(results, truncate, output_dir, rollout_id)

    # Show billing
    billing = data.get("billing", {})
    tokens = billing.get("prompt_tokens", 0) + billing.get("completion_tokens", 0)
    if tokens > 0:
        click.echo(f"Tokens: {tokens}")

    # Fail if any rollouts failed
    if failed_count > 0:
        raise click.ClickException(f"{failed_count} rollout(s) failed. See errors above.")


def _store_rollout_id(rollout_id: str, data: dict):
    """Store rollout ID and results in ./rollouts/<id>.txt"""
    rollouts_dir = Path("rollouts")
    rollouts_dir.mkdir(exist_ok=True)

    filepath = rollouts_dir / f"{rollout_id}.txt"
    with open(filepath, "w") as f:
        f.write(f"Rollout ID: {rollout_id}\n")
        f.write(f"Status: {data.get('status', 'unknown')}\n")
        f.write(f"S3 Path: rollouts/{rollout_id}/result.json\n")
        f.write("\n")

        # Write summary
        results = data.get("results", [])
        successful = [r for r in results if r.get("success")]
        if successful:
            rewards = [r.get("total_reward", 0) for r in successful]
            f.write(f"Successful: {len(successful)}/{len(results)}\n")
            f.write(f"Mean Reward: {sum(rewards) / len(rewards):.3f}\n")

        # Write billing
        billing = data.get("billing", {})
        tokens = billing.get("prompt_tokens", 0) + billing.get("completion_tokens", 0)
        if tokens > 0:
            f.write(f"Tokens: {tokens}\n")

        f.write("\n--- Full Results ---\n")
        f.write(json.dumps(data, indent=2))

    click.echo(f"Stored: {click.style(str(filepath), fg=TEAL_RGB)}")


def _display_results(
    results: list[dict],
    truncate: int | None,
    output_dir: Path | None,
    rollout_id: str | None = None,
) -> int:
    """Display rollout results.

    Returns:
        Number of failed rollouts (0 = all succeeded)
    """
    rewards = []
    failed_count = 0

    for idx, result in enumerate(results):
        click.echo(f"Rollout {idx + 1}/{len(results)}")

        if not result.get("success"):
            click.echo(click.style(f"  ✗ {result.get('error', 'Unknown error')}", fg="red"))
            click.echo()
            failed_count += 1
            continue

        total_reward = result.get("total_reward", 0.0)
        rewards.append(total_reward)

        # Show conversation
        for msg in result.get("conversation", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if truncate and len(content) > truncate:
                content = content[:truncate] + "..."
            tag = click.style(f"[{role}]", fg="red")
            click.echo(f"  {tag} {content}")

        reward_breakdown = result.get("rewards", {})
        reward_errors = result.get("errors", [])

        # Check if any reward returned "error" or "timeout"
        has_reward_error = any(
            v in ("error", "timeout") for v in reward_breakdown.values()
        ) if reward_breakdown else False

        # Handle both float values and string values (like "timeout" or "error")
        reward_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, int | float) else f"{k}={v}"
            for k, v in reward_breakdown.items()
        )
        turns = result.get("turns", 0)

        # Show reward=0 if any reward errored
        display_reward = 0.0 if has_reward_error else total_reward
        reward_color = "red" if has_reward_error else TEAL_RGB

        click.echo(
            f"  {click.style('reward', fg=reward_color)}={display_reward:.3f} "
            f"| turns={turns} "
            f"| [{reward_str}]"
        )

        # Show reward errors
        if reward_errors:
            for err in reward_errors:
                click.echo(click.style(f"    ⚠ {err}", fg="yellow"))

        # Show metadata (ground truth, etc.)
        metadata = result.get("metadata", {})
        if metadata:
            # Show ground truth answer if present
            if "answer" in metadata:
                click.echo(f"  {click.style('expected', fg='yellow')}={metadata['answer']}")
            # Show other metadata fields (excluding internal ones)
            other_meta = {
                k: v
                for k, v in metadata.items()
                if k not in ("answer", "prompt_index", "iteration", "batch")
            }
            if other_meta:
                meta_str = ", ".join(f"{k}={v}" for k, v in other_meta.items())
                click.echo(f"  {click.style('metadata', fg='cyan')}: {meta_str}")
        click.echo()

    # Save to files if requested
    if output_dir and results:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for idx, result in enumerate(results):
            if result.get("success"):
                filename = output_dir / f"rollout_{timestamp}_{idx + 1}.json"
                filename.write_text(json.dumps(result, indent=2))
        click.echo(f"Results saved to {click.style(str(output_dir), fg=TEAL_RGB)}")

    # Summary
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        click.echo()
        click.echo(f"Mean reward: {click.style(f'{mean_reward:.3f}', fg=TEAL_RGB)}")
        if rollout_id:
            click.echo(f"Run Id: {click.style(rollout_id, dim=True)}")

    return failed_count


async def _test_async(
    project_dir: Path,
    num_rollouts: int,
    model_override: str | None,
    max_context_window_override: int | None,
    api_url: str,
    debug: bool = False,
    openai_api_key: str | None = None,
    entries: tuple[int, ...] = (),
    start_time: float | None = None,
):
    # Start timing from command invocation
    if start_time is None:
        start_time = time.time()
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

    try:
        config = ProjectConfig(**config_data)
    except Exception as e:
        # Format pydantic validation errors nicely
        error_msg = str(e)
        if "validation error" in error_msg.lower():
            import re

            # Extract field and allowed values from pydantic error
            field_match = re.search(r"(\w+\.\w+|\w+)\n\s+Input should be (.+?) \[", error_msg)
            if field_match:
                field = field_match.group(1)
                allowed = field_match.group(2)
                value_match = re.search(r"input_value='([^']+)'", error_msg)
                value = value_match.group(1) if value_match else "unknown"

                click.echo()
                click.echo(click.style("  Invalid value in config.yml", fg="cyan", bold=True))
                click.echo()
                click.echo(f"  Field:    {click.style(field, fg='white', bold=True)}")
                click.echo(f"  Got:      {click.style(value, fg='red')}")
                click.echo(f"  Expected: {click.style(allowed, fg='green')}")
                click.echo()
                raise SystemExit(1)
        raise click.ClickException(f"Failed to parse config.yml: {e}")

    is_distill = config.dataset_type.value == "distill"
    is_sft = config.dataset_type.value == "sft"
    is_midtrain = config.dataset_type.value == "midtrain"

    rewards_path = project_dir / "rewards.py"
    tools_path = project_dir / "tools.py"
    train_path = project_dir / "train.jsonl"

    # rewards.py is required for RL, optional for distillation/SFT/midtrain
    if not rewards_path.exists() and not is_distill and not is_sft and not is_midtrain:
        raise click.ClickException("rewards.py not found in project directory")
    if not train_path.exists():
        raise click.ClickException("train.jsonl not found in project directory")

    # Read user code files to send to the API
    rewards_py_code = rewards_path.read_text() if rewards_path.exists() else None
    tools_py_code = tools_path.read_text() if tools_path.exists() else None

    # Read requirements.txt if exists
    requirements_path = project_dir / "requirements.txt"
    requirements_txt = requirements_path.read_text() if requirements_path.exists() else None
    if requirements_txt:
        click.echo("  Found requirements.txt")

    # Load samples
    samples = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]

    # Read Dockerfile.* files for local/ docker images
    dockerfiles: dict[str, str] = {}
    for dockerfile_path in project_dir.glob("Dockerfile.*"):
        dockerfiles[dockerfile_path.name] = dockerfile_path.read_text()
        click.echo(f"  Found {dockerfile_path.name}")

    # Read secrets from .env file and environment variables
    # Environment variables override .env values
    project_secrets: dict[str, str] = {}
    env_path = project_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                # Remove quotes if present
                value = value.strip().strip("'\"")
                project_secrets[key.strip()] = value

    # Also check common secret env vars from shell (override .env if set)
    secret_patterns = [
        "API_KEY",
        "SECRET",
        "TOKEN",
        "PROJECT_ID",
        "BROWSERBASE",
        "OPENAI",
        "GEMINI",
        "HF_",
    ]
    for key, value in os.environ.items():
        if any(pattern in key for pattern in secret_patterns):
            project_secrets[key] = value

    if project_secrets:
        click.echo(f"  Loaded secrets: {list(project_secrets.keys())}")

    if not samples:
        raise click.ClickException("train.jsonl is empty")

    # Model selection: --model flag overrides default (gpt-5-nano)
    model_name = model_override if model_override else "gpt-5-nano"

    # Validate model - only allow gpt-5 models or GPU models
    valid_gpt5_models = {"gpt-5-nano", "gpt-5-mini", "gpt-5.2", "gpt-5-pro"}
    is_gpt5_model = model_name in valid_gpt5_models or model_name.startswith("gpt-5")
    is_gpu = is_gpu_model(model_name)
    is_model_id_check = _looks_like_model_id(model_name)

    if not is_gpt5_model and not is_gpu and not is_model_id_check:
        click.echo()
        click.echo(click.style("  Unsupported Model", fg="red", bold=True))
        click.echo()
        click.echo(f"  Model '{model_name}' is not supported for rnow test.")
        click.echo()
        click.echo("  Supported models:")
        click.echo(click.style("    • gpt-5-nano (fastest, recommended for testing)", dim=True))
        click.echo(click.style("    • gpt-5-mini", dim=True))
        click.echo(click.style("    • gpt-5.2", dim=True))
        click.echo(click.style("    • gpt-5-pro (highest quality)", dim=True))
        click.echo(click.style("    • GPU models (Qwen/Qwen3-8B, etc.)", dim=True))
        click.echo(click.style("    • Finetuned model IDs", dim=True))
        click.echo()
        raise SystemExit(1)

    # Get rollout settings from config (--max-context-window overrides config)
    max_context_window = (
        max_context_window_override
        if max_context_window_override
        else (config.rollout.max_context_window if config.rollout else 32768)
    )
    max_turns = config.rollout.max_turns if config.rollout else 1
    termination_policy = config.rollout.termination_policy if config.rollout else "last_tool"
    max_tool_response = config.rollout.max_tool_response if config.rollout else None
    mcp_url = config.rollout.mcp_url if config.rollout else None

    # Detect model type
    use_gpu = is_gpu_model(model_name)
    is_model_id = _looks_like_model_id(model_name)

    # Check for OpenAI API key (required for OpenAI models)
    if not use_gpu and not openai_api_key:
        click.echo()
        click.echo(click.style("  Missing API Key", fg="red", bold=True))
        click.echo()
        click.echo(
            click.style("  We default to OpenAI models for lower latency during testing.", dim=True)
        )
        click.echo(
            click.style(
                "  Use --model to specify a different model (e.g., --model Qwen/Qwen3-8B)", dim=True
            )
        )
        click.echo()
        click.echo(
            f"  {click.style('OPENAI_API_KEY', fg=TEAL_RGB)} environment variable is required."
        )
        click.echo()
        click.echo("  Set it in your " + click.style(".env", fg=TEAL_RGB) + " file:")
        click.echo(click.style("    OPENAI_API_KEY=sk-...", dim=True))
        click.echo()
        raise SystemExit(1)

    # Display model info
    if use_gpu:
        if is_model_id:
            click.echo(f"Model Id: {click.style(model_name, dim=True)}")
        else:
            click.echo(
                f"Model: {click.style(model_name, fg=TEAL_RGB)} {click.style('(GPU)', dim=True)}"
            )
    else:
        click.echo(
            f"Model: {click.style(model_name, fg=TEAL_RGB)} {click.style('(OpenAI API)', dim=True)}"
        )

    # Note for distillation/SFT/midtrain mode
    if is_distill or is_sft or is_midtrain:
        click.echo(
            click.style("Note: ", dim=True)
            + "SFT/Distillation/Midtrain test only runs rollouts (no rewards)"
        )
    click.echo()

    try:
        # Create one RolloutClient for all rollouts
        client = RolloutClient(
            api_base=api_url,
            model=model_name,
            max_context_window=max_context_window,
            temperature=1.0,
            max_turns=max_turns,
            termination_policy=termination_policy,
            max_tool_response=max_tool_response,
            debug=debug,
            smoke_test=True,
            openai_api_key=openai_api_key,
            mcp_url=mcp_url,
        )

        # Select samples for batch rollout
        if entries:
            # Parse entries - support both "-e 0 -e 2" and "-e 0,2,5"
            entry_indices = []
            for entry in entries:
                # Handle comma-separated values
                for part in str(entry).split(","):
                    part = part.strip()
                    if part:
                        try:
                            idx = int(part)
                        except ValueError:
                            raise click.ClickException(f"Invalid entry index: {part}")
                        if idx < 0 or idx >= len(samples):
                            raise click.ClickException(
                                f"Entry index {idx} out of range. train.jsonl has {len(samples)} entries (0-{len(samples) - 1})"
                            )
                        entry_indices.append(idx)

            if not entry_indices:
                raise click.ClickException("No valid entry indices provided")

            selected_samples = [samples[idx] for idx in entry_indices]
            click.echo(f"Testing entries: {entry_indices}")
        else:
            # Random selection
            selected_samples = [random.choice(samples) for _ in range(num_rollouts)]

        try:
            # Start rollout and stream results
            _, batch_results = await client.run_batch_rollouts(
                samples=selected_samples,
                tools_py_code=tools_py_code,
                rewards_py_code=rewards_py_code,
                requirements_txt=requirements_txt,
                dockerfiles=dockerfiles if dockerfiles else None,
                secrets=project_secrets if project_secrets else None,
                timeout_minutes=60,
                start_time=start_time,
            )
        except asyncio.CancelledError:
            batch_results = []

        # Check if shutdown was requested
        if _shutdown_requested:
            await client.close()
            return

        # Show summary (results already streamed)
        failed_count = len([r for r in batch_results if not r.get("success")])

        # Close client
        await client.close()

    except Exception:
        raise

    # Show completion
    click.echo(click.style("Rollout Complete", fg=TEAL_RGB, bold=True))

    # Fail if any rollouts failed
    if failed_count > 0:
        raise click.ClickException(f"{failed_count} rollout(s) failed. See errors above.")
