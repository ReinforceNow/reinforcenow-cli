# rnow/cli/eval.py
"""
Eval command for running model evaluations and calculating pass@k metrics.

Uses server-side execution: CLI triggers Modal which runs rollouts and
calculates pass@k directly, saving results to the database.

pass@k is calculated using the unbiased estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples, k = samples to consider.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import click
import httpx
import yaml
from rich.console import Console

from rnow.cli.auth import get_auth_headers
from rnow.cli.common import get_active_organization
from rnow.cli.cube import CubeSpinner
from rnow.cli.test import DEFAULT_API_URL, TEAL_RGB, is_gpu_model
from rnow.cli.upload import get_upload_session, upload_with_boto3
from rnow.models import ProjectConfig

# Global flag for graceful shutdown
_shutdown_requested = False

console = Console()


@click.command(name="eval")
@click.option(
    "--dir",
    "-d",
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    help="Project directory containing config.yml, rewards.py, tools.py, train.jsonl",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to evaluate. Defaults to model.path from config.yml. Can be a finetuned model ID or base model name.",
)
@click.option(
    "--pass1/--no-pass1",
    default=True,
    help="Calculate pass@1 metric (default: enabled)",
)
@click.option(
    "--pass4/--no-pass4",
    default=False,
    help="Calculate pass@4 metric (default: disabled)",
)
@click.option(
    "--pass8/--no-pass8",
    default=False,
    help="Calculate pass@8 metric (default: disabled)",
)
@click.option(
    "--max-samples",
    "-n",
    default=None,
    type=int,
    help="Maximum number of samples to evaluate (default: all samples)",
)
@click.option(
    "--project-id",
    default=None,
    help="Project ID to associate with this evaluation (required)",
)
@click.option(
    "--eval-id",
    "source_eval_id",
    default=None,
    help="Re-run an existing eval with a different model. Reuses all files (train.jsonl, rewards.py, tools.py) from the source eval.",
)
@click.option(
    "--pass1-temp",
    default=None,
    type=float,
    help="Temperature for pass@1 (default: 0.2). Not supported by gpt-5-mini, gpt-5-nano, gpt-5.2-pro.",
)
@click.option(
    "--pass4-temp",
    default=None,
    type=float,
    help="Temperature for pass@4 (default: 0.6). Not supported by gpt-5-mini, gpt-5-nano, gpt-5.2-pro.",
)
@click.option(
    "--pass8-temp",
    default=None,
    type=float,
    help="Temperature for pass@8 (default: 0.8). Not supported by gpt-5-mini, gpt-5-nano, gpt-5.2-pro.",
)
@click.option(
    "--reasoning-mode",
    default=None,
    type=click.Choice(["disabled", "none", "minimal", "low", "medium", "high", "xhigh"]),
    help="Reasoning effort level. Use 'disabled'/'none' to turn off reasoning.",
)
@click.option(
    "--max-turns",
    default=None,
    type=int,
    help="Max conversation turns per rollout. Overrides config.yml rollout.max_turns.",
)
@click.option(
    "--max-context-window",
    default=None,
    type=int,
    help="Max context window in tokens. Overrides config.yml rollout.max_context_window.",
)
@click.option(
    "--termination-policy",
    default=None,
    type=click.Choice(["last_tool", "max_turns"]),
    help="When to end rollout. Overrides config.yml rollout.termination_policy.",
)
@click.option(
    "--max-tool-response",
    default=None,
    type=int,
    help="Max characters in tool response. Overrides config.yml rollout.max_tool_response.",
)
@click.option(
    "--mcp-url",
    default=None,
    help="MCP server URL for browser automation. Overrides config.yml rollout.mcp_url.",
)
@click.option(
    "--max-billing",
    default=None,
    type=float,
    help="Max cost in dollars for this eval. Overrides config.yml trainer.max_billing.",
)
@click.option(
    "--api-url",
    envvar="RNOW_API_URL",
    default=None,
    hidden=True,
    help="Base URL of the Next.js backend",
)
@click.pass_context
def eval_cmd(
    ctx,
    project_dir,
    model,
    pass1,
    pass4,
    pass8,
    max_samples,
    project_id,
    source_eval_id,
    pass1_temp,
    pass4_temp,
    pass8_temp,
    reasoning_mode,
    max_turns,
    max_context_window,
    termination_policy,
    max_tool_response,
    mcp_url,
    max_billing,
    api_url,
):
    """Run model evaluation and calculate pass@k metrics.

    Runs rollouts on all samples in train.jsonl and calculates pass@1, pass@4,
    and/or pass@8 metrics using the unbiased estimator from the Codex paper.
    Results are saved to the rnow database.

    Example:
        rnow eval --model cm123abc --project-id proj456 --pass1 --pass8
        rnow eval --eval-id cmxyz123 --model Qwen/Qwen3-8B --pass1 --pass4
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Validate at least one metric is enabled
    if not pass1 and not pass4 and not pass8:
        raise click.ClickException("At least one pass@k metric must be enabled")

    # Start timing immediately
    start_time = time.time()

    # Load .env file into os.environ
    env_file = project_dir / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)

    resolved_api_url = api_url or ctx.obj.get("api_url", "").replace("/api", "") or DEFAULT_API_URL

    # Get OpenAI API key (may not be needed for GPU models)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    async def run_with_cancellation():
        """Run eval with proper cancellation support."""
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
            await _eval_async(
                project_dir=project_dir,
                model=model,
                pass1=pass1,
                pass4=pass4,
                pass8=pass8,
                max_samples=max_samples,
                project_id=project_id,
                source_eval_id=source_eval_id,
                pass1_temp=pass1_temp,
                pass4_temp=pass4_temp,
                pass8_temp=pass8_temp,
                reasoning_mode=reasoning_mode,
                max_turns=max_turns,
                max_context_window=max_context_window,
                termination_policy=termination_policy,
                max_tool_response=max_tool_response,
                mcp_url=mcp_url,
                max_billing=max_billing,
                api_url=resolved_api_url,
                openai_api_key=openai_api_key,
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


async def _eval_async(
    project_dir: Path,
    model: str,
    pass1: bool,
    pass4: bool,
    pass8: bool,
    max_samples: int | None,
    project_id: str,
    api_url: str,
    source_eval_id: str | None = None,
    pass1_temp: float | None = None,
    pass4_temp: float | None = None,
    pass8_temp: float | None = None,
    reasoning_mode: str | None = None,
    max_turns: int | None = None,
    max_context_window: int | None = None,
    termination_policy: str | None = None,
    max_tool_response: int | None = None,
    mcp_url: str | None = None,
    max_billing: float | None = None,
    openai_api_key: str | None = None,
    start_time: float | None = None,
):
    """Run the evaluation asynchronously via cloud endpoint."""
    global _shutdown_requested

    if start_time is None:
        start_time = time.time()

    project_dir = Path(project_dir)

    # Validate temperature + reasoning_mode compatibility for OpenAI models
    # Per OpenAI docs:
    #   gpt-5-nano, gpt-5-mini, gpt-5.2-pro: NEVER accept temperature
    #   gpt-5.2: accepts temperature ONLY with reasoning effort "none" (the default)
    _NO_TEMP_MODELS = {"gpt-5-mini", "gpt-5-nano", "gpt-5.2-pro"}
    _VALID_REASONING: dict[str, set[str]] = {
        "gpt-5-nano": {"minimal", "low", "medium", "high"},
        "gpt-5-mini": {"minimal", "low", "medium", "high"},
        "gpt-5.2": {"none", "low", "medium", "high", "xhigh"},
        "gpt-5.2-pro": {"medium", "high", "xhigh"},
    }

    any_temp_set = any(t is not None for t in (pass1_temp, pass4_temp, pass8_temp))
    if model and model.startswith("gpt-"):
        if any_temp_set and model in _NO_TEMP_MODELS:
            raise click.ClickException(
                f"{model} does not support temperature overrides. "
                "It is a reasoning model that controls its own sampling."
            )
        if (
            any_temp_set
            and model == "gpt-5.2"
            and reasoning_mode
            and reasoning_mode not in ("disabled", "none")
        ):
            raise click.ClickException(
                f"{model} does not support temperature when reasoning is enabled. "
                "Remove temperature overrides or set --reasoning-mode disabled/none."
            )
        if (
            reasoning_mode
            and reasoning_mode not in ("disabled", "none")
            and model in _VALID_REASONING
            and reasoning_mode not in _VALID_REASONING[model]
        ):
            valid = ", ".join(sorted(_VALID_REASONING[model]))
            raise click.ClickException(
                f'{model} does not support --reasoning-mode "{reasoning_mode}". '
                f"Valid modes: {valid}"
            )

    # --eval-id mode: reuse files from an existing eval, only need model
    if source_eval_id:
        if model is None:
            raise click.ClickException(
                "When using --eval-id, you must specify --model to evaluate with."
            )

        # Still load .env for secrets
        project_secrets: dict[str, str] = {}
        env_path = project_dir / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip().strip("'\"")
                    project_secrets[key.strip()] = value

        # Also check common secret env vars from shell
        secret_patterns = ["API_KEY", "SECRET", "TOKEN", "PROJECT_ID", "OPENAI", "GEMINI", "HF_"]
        for key, value in os.environ.items():
            if any(pattern in key for pattern in secret_patterns):
                project_secrets[key] = value

        use_gpu = is_gpu_model(model)
        if not use_gpu and not openai_api_key:
            raise click.ClickException(
                "OPENAI_API_KEY required for OpenAI models. "
                "Set it in .env or use a GPU model (e.g., --model Qwen/Qwen3-8B)"
            )

        # Determine max k value needed
        max_k = 1
        if pass4:
            max_k = max(max_k, 4)
        if pass8:
            max_k = max(max_k, 8)

        # Build metrics string for display
        metrics_str = []
        if pass1:
            metrics_str.append("pass@1")
        if pass4:
            metrics_str.append("pass@4")
        if pass8:
            metrics_str.append("pass@8")

        # Get auth headers
        auth_headers = get_auth_headers()
        if not auth_headers:
            raise click.ClickException("Not logged in. Run 'rnow login' first.")

        organization_id = get_active_organization()
        if not organization_id:
            raise click.ClickException(
                "No organization selected. Run "
                + click.style("rnow orgs", fg=TEAL_RGB)
                + " to select one."
            )

        if project_secrets:
            click.echo(
                click.style(
                    f"🔐 Loaded {len(project_secrets)} secret(s) from .env + environment", dim=True
                )
            )

        click.echo(click.style("Source eval: ", fg=TEAL_RGB) + f"{source_eval_id}")
        click.echo()

        # Start cube spinner
        spinner = CubeSpinner()
        spinner.start()

        request_payload = {
            "modelId": model,
            "projectId": project_id,
            "organizationId": organization_id,
            "sourceEvalId": source_eval_id,
            "pass1": pass1,
            "pass4": pass4,
            "pass8": pass8,
            "secrets": project_secrets if project_secrets else None,
            "model": model,
            "use_gpu": use_gpu,
        }

        # Only send overrides if explicitly set (let backend/source eval provide defaults)
        _overrides = {
            "pass1_temp": pass1_temp,
            "pass4_temp": pass4_temp,
            "pass8_temp": pass8_temp,
            "reasoning_mode": reasoning_mode,
            "max_turns": max_turns,
            "max_context_window": max_context_window,
            "termination_policy": termination_policy,
            "max_tool_response": max_tool_response,
            "mcp_url": mcp_url,
            "max_billing": max_billing,
        }
        for key, val in _overrides.items():
            if val is not None:
                request_payload[key] = val

        if max_samples:
            request_payload["num_samples"] = max_samples

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{api_url}/api/evals/run",
                    headers={**auth_headers, "Content-Type": "application/json"},
                    json=request_payload,
                )

                if response.status_code != 200:
                    spinner.stop()
                    error_text = response.text
                    try:
                        error_data = response.json()
                        error_text = error_data.get("error", error_text)
                    except Exception:
                        pass
                    raise click.ClickException(f"Failed to start evaluation: {error_text}")

                data = response.json()
                eval_id = data.get("data", {}).get("evalId")

                spinner.stop(keep_visible=True)

                click.echo(click.style("Evaluation started", fg=TEAL_RGB, bold=True))
                click.echo()
                click.echo(f"  Model:    {click.style(model, fg=TEAL_RGB)}")
                click.echo(f"  Source:   {source_eval_id}")
                click.echo(f"  Metrics:  {', '.join(metrics_str)}")
                click.echo()
                click.echo("View your evaluation:")
                click.echo(click.style(f"https://www.reinforcenow.ai/evals/{eval_id}", fg=TEAL_RGB))

            except httpx.RequestError as e:
                spinner.stop()
                raise click.ClickException(f"Failed to connect to API: {e}")

        return  # Done with --eval-id path

    # Standard path: load from project directory
    # Load config
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
        raise click.ClickException(f"Failed to parse config: {e}")

    # Get model from config if not provided via flag
    if model is None:
        if config.model and config.model.path:
            model = config.model.path
        else:
            raise click.ClickException(
                "No model specified. Use --model flag or set model.path in config.yml"
            )

    # Get project_id from config if not provided via flag, or generate one
    if project_id is None:
        if config.project_id:
            project_id = config.project_id
        else:
            # Generate new project_id and save to config (like rnow init)
            import uuid

            project_id = str(uuid.uuid4())
            config_data["project_id"] = project_id
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            click.echo(click.style(f"Generated project_id: {project_id}", dim=True))

    # Load project files
    rewards_path = project_dir / "rewards.py"
    tools_path = project_dir / "tools.py"
    train_path = project_dir / "train.jsonl"

    if not train_path.exists():
        raise click.ClickException("train.jsonl not found in project directory")

    rewards_py_code = rewards_path.read_text() if rewards_path.exists() else None
    tools_py_code = tools_path.read_text() if tools_path.exists() else None

    # Read requirements.txt if exists
    requirements_path = project_dir / "requirements.txt"
    requirements_txt = requirements_path.read_text() if requirements_path.exists() else None

    # Load samples
    samples = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]

    if not samples:
        raise click.ClickException("train.jsonl is empty")

    # Limit samples if requested
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    # For large datasets, upload to S3 first (like rnow run does)
    # Calculate actual payload size after any max_samples limiting
    samples_json = "\n".join(json.dumps(s) for s in samples)
    samples_size = len(samples_json.encode("utf-8"))

    # 4MB threshold to stay under API payload limits
    MAX_INLINE_BYTES = 4 * 1024 * 1024
    s3_dataset_key = None

    if samples_size > MAX_INLINE_BYTES:
        try:
            import tempfile
            import uuid

            # Generate a version ID for this eval upload
            eval_version_id = f"eval-{uuid.uuid4().hex[:8]}"

            # Write samples to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp.write(samples_json)
                tmp_path = Path(tmp.name)

            # Get S3 upload session (needs /api suffix)
            upload_session = get_upload_session(f"{api_url}/api", project_id, eval_version_id)

            # Upload to S3
            upload_with_boto3(upload_session, [("train.jsonl", tmp_path, "project")])
            tmp_path.unlink()  # Clean up temp file

            # Build the S3 key that was uploaded
            s3_dataset_key = f"{upload_session['projectPrefix']}/train.jsonl"
            click.echo(
                click.style(
                    f"Uploaded dataset to S3 ({samples_size / 1024 / 1024:.1f} MB)", dim=True
                )
            )
        except Exception as e:
            raise click.ClickException(f"Failed to upload large dataset: {e}")

    # Read Dockerfile.* files
    dockerfiles: dict[str, str] = {}
    for dockerfile_path in project_dir.glob("Dockerfile.*"):
        dockerfiles[dockerfile_path.name] = dockerfile_path.read_text()

    # Read secrets from .env
    project_secrets: dict[str, str] = {}
    env_path = project_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                project_secrets[key.strip()] = value

    # Also check common secret env vars from shell
    secret_patterns = ["API_KEY", "SECRET", "TOKEN", "PROJECT_ID", "OPENAI", "GEMINI", "HF_"]
    for key, value in os.environ.items():
        if any(pattern in key for pattern in secret_patterns):
            project_secrets[key] = value

    # Warn if HF_TOKEN is not set (HuggingFace rate-limits unauthenticated requests)
    if not project_secrets.get("HF_TOKEN") and not os.environ.get("HF_TOKEN"):
        click.echo(
            click.style(
                "Warning: HF_TOKEN not found. Add HF_TOKEN=hf_xxx to your .env file to avoid "
                "HuggingFace rate limiting. Get a token at https://huggingface.co/settings/tokens",
                fg="red",
            )
        )

    # Determine max k value needed (number of rollouts per sample)
    max_k = 1
    if pass4:
        max_k = max(max_k, 4)
    if pass8:
        max_k = max(max_k, 8)

    # Validate model
    use_gpu = is_gpu_model(model)
    if not use_gpu and not openai_api_key:
        raise click.ClickException(
            "OPENAI_API_KEY required for OpenAI models. "
            "Set it in .env or use a GPU model (e.g., --model Qwen/Qwen3-8B)"
        )

    # Get rollout settings from config, CLI flags override
    if max_context_window is None:
        max_context_window = config.rollout.max_context_window if config.rollout else 32768
    if max_turns is None:
        max_turns = config.rollout.max_turns if config.rollout else 1
    if termination_policy is None:
        termination_policy = config.rollout.termination_policy if config.rollout else "last_tool"
    if max_tool_response is None:
        max_tool_response = config.rollout.max_tool_response if config.rollout else None
    if mcp_url is None:
        mcp_url = config.rollout.mcp_url if config.rollout else None
    if reasoning_mode is None:
        reasoning_mode = config.rollout.reasoning_mode if config.rollout else None

    # Build metrics string for display
    metrics_str = []
    if pass1:
        metrics_str.append("pass@1")
    if pass4:
        metrics_str.append("pass@4")
    if pass8:
        metrics_str.append("pass@8")

    # Get auth headers
    auth_headers = get_auth_headers()
    if not auth_headers:
        raise click.ClickException("Not logged in. Run 'rnow login' first.")

    # Get organization from CLI setting (same as rnow run)
    organization_id = get_active_organization()
    if not organization_id:
        raise click.ClickException(
            "No organization selected. Run "
            + click.style("rnow orgs", fg=TEAL_RGB)
            + " to select one."
        )

    # Resolve model ID - if it's a UUID, use it directly; otherwise use the model name
    model_id = model

    # Display secrets info (like rnow run)
    if project_secrets:
        click.echo(
            click.style(
                f"🔐 Loaded {len(project_secrets)} secret(s) from .env + environment", dim=True
            )
        )

    # Display tokenizer info
    click.echo(click.style("Loading tokenizer...", dim=True), nl=False)
    try:
        from rnow.cli.token_count import get_tokenizer_for_model

        tokenizer_info = get_tokenizer_for_model(model)
        if tokenizer_info:
            tokenizer_type = tokenizer_info[0]
            label = "Harmony" if tokenizer_type == "harmony" else "HuggingFace"
            click.echo(
                "\r" + click.style("Tokenizer: ", fg=TEAL_RGB) + f"{label} ({model})" + " " * 10
            )
        else:
            click.echo(
                "\r"
                + click.style("Tokenizer: ", fg="yellow")
                + "not available, using estimates"
                + " " * 10
            )
    except Exception:
        click.echo(
            "\r"
            + click.style("Tokenizer: ", fg="yellow")
            + "not available, using estimates"
            + " " * 10
        )

    # Display MCP/tools info
    has_mcp_url = mcp_url is not None
    has_tools_py = tools_py_code is not None
    if has_mcp_url and has_tools_py:
        click.echo(click.style("Tools: ", fg=TEAL_RGB) + f"Using MCP ({mcp_url}) and tools.py")
    elif has_mcp_url:
        click.echo(click.style("Tools: ", fg=TEAL_RGB) + f"Using MCP ({mcp_url})")
    elif has_tools_py:
        click.echo(click.style("Tools: ", fg=TEAL_RGB) + "Using tools.py tools")

    # Display context window info
    click.echo(
        click.style("Context: ", fg=TEAL_RGB)
        + f"{max_context_window:,} tokens (max_turns={max_turns})"
    )

    click.echo()

    # Start cube spinner
    spinner = CubeSpinner()
    spinner.start()

    # Call the /api/evals/run endpoint
    # Build request payload - use S3 key for large files, inline samples for small ones
    request_payload = {
        "modelId": model_id,
        "projectId": project_id,
        "organizationId": organization_id,
        "pass1": pass1,
        "pass4": pass4,
        "pass8": pass8,
        # Rollout data - either S3 key or inline samples
        "s3_dataset_key": s3_dataset_key,
        "samples": None if s3_dataset_key else samples,
        "num_samples": len(samples),  # Always send count for display
        "rewards_py_code": rewards_py_code,
        "tools_py_code": tools_py_code,
        "requirements_txt": requirements_txt,
        "dockerfiles": dockerfiles if dockerfiles else None,
        "secrets": project_secrets if project_secrets else None,
        # Config
        "model": model,
        "max_context_window": max_context_window,
        "max_turns": max_turns,
        "termination_policy": termination_policy,
        "max_tool_response": max_tool_response,
        "mcp_url": mcp_url,
        "reasoning_mode": reasoning_mode,
        "use_gpu": use_gpu,
        "max_billing": max_billing,
    }

    # Only send per-k temperatures if explicitly set (let backend auto-select based on pass@k)
    if pass1_temp is not None:
        request_payload["pass1_temp"] = pass1_temp
    if pass4_temp is not None:
        request_payload["pass4_temp"] = pass4_temp
    if pass8_temp is not None:
        request_payload["pass8_temp"] = pass8_temp

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{api_url}/api/evals/run",
                headers={**auth_headers, "Content-Type": "application/json"},
                json=request_payload,
            )

            if response.status_code != 200:
                spinner.stop()
                error_text = response.text
                try:
                    error_data = response.json()
                    error_text = error_data.get("error", error_text)
                    # Check for organization mismatch error
                    if error_data.get("code") == "ORG_MISMATCH":
                        click.echo()
                        click.echo(
                            click.style("Error: ", fg="red", bold=True)
                            + "This model belongs to a different organization."
                        )
                        click.echo()
                        click.echo("To fix this:")
                        click.echo(
                            "  1. Go to "
                            + click.style("https://www.reinforcenow.ai/settings", fg=TEAL_RGB)
                        )
                        click.echo("  2. Switch to the organization that owns this model")
                        click.echo(
                            "  3. Run "
                            + click.style("rnow login", fg=TEAL_RGB)
                            + " again to refresh your session"
                        )
                        click.echo("  4. Then retry this command")
                        raise SystemExit(1)
                except click.ClickException:
                    raise
                except SystemExit:
                    raise
                except Exception:
                    pass
                raise click.ClickException(f"Failed to start evaluation: {error_text}")

            data = response.json()
            eval_id = data.get("data", {}).get("evalId")

            # Stop spinner but keep cube visible on success
            spinner.stop(keep_visible=True)

            # Show summary (similar to rnow run)
            total_rollouts = len(samples) * max_k
            click.echo(click.style("Evaluation started", fg=TEAL_RGB, bold=True))
            click.echo()
            click.echo(f"  Model:    {click.style(model, fg=TEAL_RGB)}")
            click.echo(f"  Samples:  {len(samples)}")
            click.echo(f"  Rollouts: {total_rollouts} ({max_k} per sample)")
            click.echo(f"  Metrics:  {', '.join(metrics_str)}")
            click.echo()
            click.echo("View your evaluation:")
            click.echo(click.style(f"https://www.reinforcenow.ai/evals/{eval_id}", fg=TEAL_RGB))

        except httpx.RequestError as e:
            spinner.stop()
            raise click.ClickException(f"Failed to connect to API: {e}")


# Alias for the command (eval is a Python builtin)
eval = eval_cmd
