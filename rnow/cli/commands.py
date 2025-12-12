# reinforcenow/cli/commands.py

import itertools
import json
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import click
import requests
import yaml
from pydantic import ValidationError

# ReinforceNow teal: #14B8A6
TEAL_RGB = (20, 184, 166)


class Spinner:
    """Simple spinner for CLI feedback."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = ""):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r\033[K{frame} {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)


from rnow import models
from rnow.cli import auth
from rnow.cli.blob import MAX_INLINE_BYTES, maybe_upload_to_blob
from rnow.cli.common import get_active_organization, require_auth
from rnow.cli.cube import CubeSpinner

CONFIG_DOCS_URL = "https://reinforcenow.ai/docs/cli-reference/configuration"


def format_validation_error(e: ValidationError) -> str:
    """
    Format Pydantic ValidationError into a user-friendly message.
    """
    lines = ["", click.style("✗ Invalid config.yml", fg="red", bold=True), ""]

    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_type = error["type"]

        # For root-level validation errors, show a more specific field name
        if not loc and error_type == "value_error":
            if "qlora_rank" in msg:
                loc = "model.qlora_rank"
            elif "batch_size" in msg:
                loc = "data"

        lines.append(f"  Field: {click.style(loc or '(root)', bold=True)}")

        # Get the input value if available (skip for dict/complex types)
        if "input" in error:
            input_val = error["input"]
            # Skip showing full config dicts
            if isinstance(input_val, dict) and len(input_val) > 3:
                pass  # Don't show large dicts
            elif isinstance(input_val, str) and len(input_val) > 50:
                lines.append(f"    Got: {repr(input_val[:50] + '...')}")
            else:
                lines.append(f"    Got: {repr(input_val)}")

        # Format the error message nicely
        if error_type == "literal_error":
            # Extract expected values from the message
            lines.append(f"    Error: {msg}")
        elif error_type == "extra_forbidden":
            lines.append("    Error: Unknown field (typo?)")
        elif error_type == "missing":
            lines.append("    Error: Required field is missing")
        elif error_type == "greater_than" or error_type == "greater_than_equal":
            lines.append(f"    Error: {msg}")
        elif error_type == "less_than_equal":
            lines.append(f"    Error: {msg}")
            if "batch_size" in loc:
                lines.append("    Hint: Maximum batch_size is 32")
            elif "group_size" in loc:
                lines.append("    Hint: Maximum group_size is 64")
        elif error_type == "value_error" and "batch_size * group_size" in msg:
            lines.append(f"    Error: {msg}")
            lines.append("    Hint: Reduce batch_size or group_size to stay within the 2048 limit")
        elif error_type == "value_error" and "qlora_rank" in msg:
            # Clean up the error message (remove "Value error, " prefix)
            clean_msg = msg.replace("Value error, ", "")
            lines.append(f"    Error: {clean_msg}")
            lines.append(
                "    Hint: Different models have different max LoRA ranks (32, 64, or 128)"
            )
        else:
            lines.append(f"    Error: {msg}")

        lines.append("")

    lines.append(f"  See: {click.style(CONFIG_DOCS_URL, fg=TEAL_RGB, underline=True)}")
    lines.append("")

    return "\n".join(lines)


def get_rewards_referenced_in_jsonl(path: Path) -> set[str]:
    """
    Extract all reward names referenced in train.jsonl.

    Scans the entire file to ensure all reward references are captured.

    Returns:
        Set of reward names referenced in the 'rewards' field across all samples.
    """
    rewards = set()

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                    if isinstance(record, dict) and "rewards" in record:
                        record_rewards = record["rewards"]
                        if isinstance(record_rewards, list):
                            rewards.update(record_rewards)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return rewards


def validate_reward_references(train_jsonl_path: Path, rewards_py_path: Path) -> list[str]:
    """
    Validate that all reward names in train.jsonl exist in rewards.py.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    try:
        from rnow.core.reward import get_reward_names_from_file
    except ImportError:
        return []  # Skip validation if module not available

    # Get rewards defined in rewards.py
    defined_rewards = get_reward_names_from_file(rewards_py_path)

    # Get rewards referenced in train.jsonl
    referenced_rewards = get_rewards_referenced_in_jsonl(train_jsonl_path)

    # Find missing rewards
    missing_rewards = referenced_rewards - defined_rewards

    if missing_rewards:
        for reward_name in sorted(missing_rewards):
            errors.append(
                f"Reward '{reward_name}' is referenced in train.jsonl but not defined in rewards.py"
            )

        if defined_rewards:
            errors.append(
                f"  Available rewards in rewards.py: {', '.join(sorted(defined_rewards))}"
            )
        else:
            errors.append("  No @reward functions found in rewards.py")

    return errors


# Import token counting utilities from dedicated module
from rnow.cli.token_count import (
    get_max_prompt_tokens,
    get_tokenizer_for_model,
)


def validate_max_tokens_for_context(
    max_tokens: int, max_prompt_tokens: int, context_window: int = models.MAX_CONTEXT_WINDOW
) -> tuple[str | None, int]:
    """
    Validate that max_tokens + max_prompt_tokens fits within context window.
    Returns (error_message, recommended_max_tokens). Error is None if valid.
    """
    total_required = max_tokens + max_prompt_tokens
    available = context_window - max_prompt_tokens
    if total_required > context_window:
        return (
            f"max_tokens ({max_tokens:,}) + prompt ({max_prompt_tokens:,}) = {total_required:,} > context window ({context_window:,})",
            available,
        )
    return None, available


def get_tools_from_env_py(env_path: Path) -> list[dict]:
    """
    Extract tool definitions from env.py as structured data.
    Returns list of tool dicts with name, description, and schema.
    """
    import ast

    if not env_path.exists():
        return []

    try:
        source = env_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    tools = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Check if function has @tool decorator
            is_tool = any(
                (isinstance(d, ast.Name) and d.id == "tool")
                or (
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "tool"
                )
                for d in node.decorator_list
            )
            if not is_tool:
                continue

            # Extract tool info
            tool = {
                "name": node.name,
                "description": ast.get_docstring(node) or "",
                "schema": {"type": "object", "properties": {}, "required": []},
            }

            # Add parameters to schema
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg not in ("self", "cls"):
                    # Try to get type annotation
                    param_type = "string"  # Default
                    if arg.annotation and isinstance(arg.annotation, ast.Name):
                        type_name = arg.annotation.id.lower()
                        if type_name in ("int", "integer"):
                            param_type = "integer"
                        elif type_name in ("float", "number"):
                            param_type = "number"
                        elif type_name in ("bool", "boolean"):
                            param_type = "boolean"
                        elif type_name in ("list", "array"):
                            param_type = "array"
                        elif type_name in ("dict", "object"):
                            param_type = "object"

                    tool["schema"]["properties"][arg.arg] = {"type": param_type}

                    # Check if it's a required arg (no default)
                    if arg in node.args.args:
                        idx = node.args.args.index(arg)
                        num_defaults = len(node.args.defaults)
                        num_args = len(node.args.args)
                        if idx < num_args - num_defaults:
                            tool["schema"]["required"].append(arg.arg)

            tools.append(tool)

    return tools


def fetch_mcp_tool_schemas(
    mcp_urls: list[str] | str | None, timeout: float = 15.0
) -> tuple[list[dict], str | None]:
    """
    Fetch tool schemas from MCP servers.

    Args:
        mcp_urls: MCP server URL(s)
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (list of tool dicts, error_message or None)
        Returns ([], error_message) if fetch fails.
    """
    if not mcp_urls:
        return [], None

    urls = mcp_urls if isinstance(mcp_urls, list) else [mcp_urls]
    all_tools = []
    error_msg = None

    try:
        from fastmcp import Client
    except ImportError:
        return [], "fastmcp not installed"

    import asyncio

    async def fetch_tools():
        nonlocal all_tools, error_msg

        # Build FastMCP config
        fastmcp_config = {"mcpServers": {}}
        for i, url in enumerate(urls):
            server_name = f"mcp_{i}"
            fastmcp_config["mcpServers"][server_name] = {"url": url}

        try:
            client = Client(fastmcp_config)
            async with client:
                tools = await client.list_tools()

                for tool in tools:
                    name = tool.name
                    description = getattr(tool, "description", "") or ""

                    # Get input schema
                    input_schema = {}
                    if hasattr(tool, "inputSchema"):
                        input_schema = tool.inputSchema
                    elif hasattr(tool, "input_schema"):
                        input_schema = tool.input_schema

                    all_tools.append(
                        {
                            "name": name,
                            "description": description,
                            "schema": input_schema,
                        }
                    )

        except Exception as e:
            error_msg = str(e)
            return False
        return True

    # Run async fetch with timeout
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(asyncio.wait_for(fetch_tools(), timeout=timeout))
        finally:
            loop.close()

        if not success:
            return [], error_msg or "connection failed"

    except asyncio.TimeoutError:
        return [], f"timeout after {timeout}s"
    except Exception as e:
        return [], str(e)

    return all_tools, None


def validate_train_jsonl(
    path: Path, dataset_type: models.DatasetType, sample_size: int = 50
) -> list[str]:
    """
    Validate train.jsonl format by sampling first N lines.
    Returns a list of error messages (empty if valid).
    """
    errors = []

    try:
        with open(path, encoding="utf-8") as f:
            lines_checked = 0
            for line_num, line in enumerate(f, start=1):
                # Skip empty lines
                stripped = line.strip()
                if not stripped:
                    continue

                # Try to parse as JSON
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e.msg}")
                    if len(errors) >= 5:
                        errors.append("... (stopping after 5 errors)")
                        return errors
                    continue

                # Check it's a dict
                if not isinstance(record, dict):
                    errors.append(
                        f"Line {line_num}: Expected JSON object, got {type(record).__name__}"
                    )
                    continue

                # Check for required 'messages' field
                if "messages" not in record:
                    errors.append(f"Line {line_num}: Missing required 'messages' field")
                    continue

                messages = record["messages"]
                if not isinstance(messages, list):
                    errors.append(f"Line {line_num}: 'messages' must be a list")
                    continue

                if len(messages) == 0:
                    errors.append(f"Line {line_num}: 'messages' list is empty")
                    continue

                # Check each message has role and content
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {line_num}: Message {msg_idx + 1} must be an object")
                        break
                    if "role" not in msg:
                        errors.append(f"Line {line_num}: Message {msg_idx + 1} missing 'role'")
                        break
                    if "content" not in msg:
                        errors.append(f"Line {line_num}: Message {msg_idx + 1} missing 'content'")
                        break
                    if msg["role"] not in ("system", "user", "assistant"):
                        errors.append(
                            f"Line {line_num}: Message {msg_idx + 1} has invalid role '{msg['role']}' (expected: system, user, assistant)"
                        )
                        break

                # For RL, check for rewards field
                if dataset_type == models.DatasetType.RL and "rewards" not in record:
                    errors.append(
                        f"Line {line_num}: Missing required 'rewards' field for RL dataset"
                    )

                # Validate optional 'tools' field if present
                if "tools" in record:
                    tools = record["tools"]
                    if not isinstance(tools, list):
                        errors.append(f"Line {line_num}: 'tools' must be a list of tool names")
                    elif not all(isinstance(t, str) for t in tools):
                        errors.append(f"Line {line_num}: 'tools' must contain only strings")

                lines_checked += 1
                if lines_checked >= sample_size:
                    break

            # Check if file was effectively empty (only whitespace)
            if lines_checked == 0:
                errors.append("File contains no valid JSON lines")

    except Exception as e:
        errors.append(f"Failed to read file: {e}")

    return errors


from functools import lru_cache


@lru_cache(maxsize=256)
def _pypi_requires_python(project: str, version: str | None = None) -> str | None:
    """
    Return the `Requires-Python` specifier for a project (and optionally a specific version),
    or None if it can't be determined.
    """
    import urllib.error
    import urllib.request

    try:
        if version:
            url = f"https://pypi.org/pypi/{project}/{version}/json"
        else:
            url = f"https://pypi.org/pypi/{project}/json"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())

        info = data.get("info", {})
        return info.get("requires_python")
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return None
    except Exception:
        return None


def check_pypi_python_compatibility(req, target_python: str = "3.11") -> str | None:
    """
    Check if a package on PyPI supports the target Python version.
    Returns an error string if clearly incompatible, otherwise None.

    Args:
        req: A packaging.requirements.Requirement object
        target_python: Target Python version string (e.g., "3.11")
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        return None

    target_version = Version(target_python)

    # Try to respect pinned version if present (foo==1.2.3)
    pinned_version = None
    for spec in req.specifier:
        if spec.operator == "==":
            pinned_version = spec.version
            break

    requires_python = _pypi_requires_python(req.name, pinned_version)
    if not requires_python:
        # Unknown compatibility → don't fail hard
        return None

    try:
        specifier = SpecifierSet(requires_python)
        if target_version not in specifier:
            if pinned_version:
                return (
                    f"Package '{req.name}=={pinned_version}' requires Python "
                    f"{requires_python}, which does not include Python {target_python}"
                )
            else:
                return (
                    f"Package '{req.name}' requires Python "
                    f"{requires_python}, which does not include Python {target_python}"
                )
    except Exception:
        return None

    return None


def validate_requirements_txt(path: Path, target_python: str = "3.11") -> list[str]:
    """
    Validate requirements.txt for format + Python compatibility.

    Checks:
    1. File is valid requirements.txt format (not TOML/other format)
    2. Each requirement line is parseable
    3. Environment markers that exclude target Python
    4. PyPI Requires-Python metadata for each package

    Returns a list of error/warning messages (empty if valid).
    """
    errors = []

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        errors.append(f"Failed to read requirements.txt: {e}")
        return errors

    # Check if it's accidentally a TOML file (common mistake)
    stripped = content.strip()
    if stripped.startswith("[") and "]" in stripped.split("\n")[0]:
        errors.append("requirements.txt appears to be in TOML format, not pip requirements format")
        errors.append(
            "Hint: requirements.txt should have one package per line, e.g., 'requests>=2.28.0'"
        )
        return errors

    # Try to parse requirements using packaging library
    try:
        from packaging.markers import default_environment
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.version import Version
    except ImportError:
        # packaging not available, skip detailed validation
        return []

    try:
        target_version = Version(target_python)
    except Exception:
        errors.append(f"Invalid target Python version: {target_python}")
        return errors

    # Environment for evaluating markers like `python_version < "3.11"`
    env = default_environment()
    env["python_version"] = f"{target_version.major}.{target_version.minor}"
    env["python_full_version"] = str(target_version)

    seen_projects: set[str] = set()

    for lineno, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Skip options like -e, --index-url, etc.
        if line.startswith("-"):
            continue

        # Try to parse as a requirement
        try:
            req = Requirement(line)
        except InvalidRequirement as e:
            errors.append(f"Line {lineno}: Invalid requirement '{line}' - {e}")
            continue

        # 1) Marker-based incompatibility (e.g., `foo; python_version < "3.11"`)
        if req.marker is not None and not req.marker.evaluate(env):
            # This requirement is explicitly excluded for Python 3.11
            errors.append(
                f"Line {lineno}: Requirement '{line}' is excluded for Python "
                f"{target_python} due to marker '{req.marker}'"
            )

        # 2) PyPI Requires-Python compatibility check (once per project)
        project_key = req.name.lower()
        if project_key not in seen_projects:
            seen_projects.add(project_key)
            compat_msg = check_pypi_python_compatibility(req, target_python)
            if compat_msg:
                errors.append(f"Line {lineno}: {compat_msg}")

    return errors


def get_thinking_mode_display(config: models.ProjectConfig) -> str:
    """Get a human-readable display string for the thinking mode."""
    thinking_mode = config.rollout.thinking_mode if config.rollout else None
    model = config.model.path

    # GPT-OSS: Reasoning models with levels
    if model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        mode_map = {
            "disabled": "Reasoning Off",
            "easy": "Reasoning Low",
            "hard": "Reasoning High",
        }
        return mode_map.get(thinking_mode, "Reasoning Medium")

    # Hybrid models: Qwen3, DeepSeek
    if model in [
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-8B-Base",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3.1-Base",
    ]:
        if thinking_mode == "disabled":
            return "Reasoning Off"
        else:
            return "Reasoning On"

    # Instruct models: no thinking support
    if model in [
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]:
        return "Reasoning Off"

    # Base Llama models
    if model in [
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-1B",
    ]:
        return "Reasoning Off"

    return "Reasoning Off"


# Simple session for API calls
session = requests.Session()
session.headers["User-Agent"] = "ReinforceNow-CLI/1.0"


def api_request(
    method: str, endpoint: str, base_url: str = None, authenticated: bool = True, **kwargs
):
    """Make API request."""
    if authenticated:
        require_auth()
        headers = kwargs.pop("headers", {})
        headers.update(auth.get_auth_headers())
        kwargs["headers"] = headers

    url = f"{base_url or 'https://www.reinforcenow.ai/api'}{endpoint}"
    return getattr(session, method)(url, **kwargs)


# ========== Auth Commands ==========


@click.command()
@click.option("--force", "-f", is_flag=True, help="Force new login even if already authenticated")
@click.pass_context
def login(ctx, force: bool):
    """Login to ReinforceNow platform.

    Uses OAuth device flow for authentication.
    """
    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")

    if not force and auth.is_authenticated():
        click.echo(click.style("✓ Already authenticated", fg="green"))
        click.echo("Use --force to re-authenticate")
        return

    # Get device code
    try:
        response = api_request(
            "post", "/auth/device/code", base_url, json={"client_id": "cli"}, authenticated=False
        )
        response.raise_for_status()
        device = models.DeviceCode(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid response from server: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to initiate login: {e}")

    # Construct the full URL with user_code parameter
    verification_url = f"{device.verification_uri}?user_code={device.user_code}"

    click.echo(f"\n{click.style('Opening browser:', fg=TEAL_RGB)} {verification_url}")
    click.echo(
        f"{click.style('Enter code:', fg=TEAL_RGB)} {click.style(device.user_code, bold=True)}\n"
    )
    webbrowser.open(verification_url)

    # Poll for token with spinner
    spinner = Spinner("Waiting for authentication...")
    spinner.start()

    start = time.time()
    try:
        while time.time() - start < device.expires_in:
            time.sleep(device.interval)

            try:
                resp = api_request(
                    "post",
                    "/auth/device/token",
                    base_url,
                    json={"device_code": device.device_code},
                    authenticated=False,
                )
                data = resp.json()
            except requests.RequestException as e:
                spinner.stop()
                raise click.ClickException(f"Network error: {e}")

            if resp.status_code == 200:
                try:
                    token = models.Token(**data)
                except ValidationError as e:
                    spinner.stop()
                    raise click.ClickException(f"Invalid token response: {e}")

                # Save credentials
                auth.DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(auth.CREDS_FILE, "w") as f:
                    json.dump(
                        {"api_key": token.access_token, "organization_id": token.organization_id}, f
                    )
                auth.CREDS_FILE.chmod(0o600)

                spinner.stop()
                click.echo(click.style("✓ Login successful!", fg="green", bold=True))
                return

            try:
                error = models.TokenError(**data)
            except ValidationError:
                spinner.stop()
                raise click.ClickException(f"Unexpected response: {data}")

            if error.error != "authorization_pending":
                spinner.stop()
                raise click.ClickException(f"Authentication failed: {error.error}")
    finally:
        spinner.stop()

    raise click.ClickException("Authentication timed out")


@click.command()
def logout():
    """Logout from ReinforceNow."""
    auth.logout()


@click.command()
@click.pass_context
def status(ctx):
    """Check authentication status and running jobs."""
    if not auth.is_authenticated():
        click.echo(click.style("✗ Not authenticated", fg="red"))
        raise click.ClickException("Run 'rnow login' to authenticate")

    click.echo(click.style("✓ Authenticated", fg=TEAL_RGB))
    org_id = get_active_organization()
    if org_id:
        click.echo(f"Organization: {org_id}")

    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")
    click.echo()
    click.echo(click.style("Running jobs:", bold=True))
    try:
        response = api_request("get", "/runs?status=running", base_url)
        response.raise_for_status()
        data = response.json()
        running_runs = data.get("data", [])
        if running_runs:
            for run in running_runs:
                run_id = run.get("id", "unknown")
                project = run.get("project", {})
                project_name = project.get("name", "Unknown project")
                phase = run.get("phase", "running")
                click.echo(f"  • {click.style(run_id, fg=TEAL_RGB)} - {project_name} ({phase})")
        else:
            click.echo("  No running jobs")
    except requests.RequestException as e:
        click.echo(click.style(f"  Error fetching runs: {e}", fg="red"))


# ========== Org Commands ==========


def _interactive_org_selector(organizations: list, active_org_id: str | None) -> str | None:
    """Interactive organization selector using arrow keys."""
    import sys

    # Find initial selection index
    selected_idx = 0
    for i, org in enumerate(organizations):
        if org.id == active_org_id:
            selected_idx = i
            break

    def render():
        lines = []
        lines.append(click.style("Select organization:", bold=True))
        lines.append("")
        for i, org in enumerate(organizations):
            is_selected = i == selected_idx
            is_active = org.id == active_org_id
            marker = "✓ " if is_active else "  "
            role = click.style(f" ({org.role.value})", dim=True)

            if is_selected:
                # Highlight selected row with teal
                prefix = click.style("› ", fg=TEAL_RGB, bold=True)
                name = click.style(f"{marker}{org.name}", fg=TEAL_RGB, bold=True)
                lines.append(f"{prefix}{name}{role}")
            else:
                prefix = "  "
                if is_active:
                    name = click.style(f"{marker}{org.name}", fg=TEAL_RGB)
                else:
                    name = f"{marker}{org.name}"
                lines.append(f"{prefix}{name}{role}")
        lines.append("")
        lines.append(click.style("↑/↓ to move, Enter to select, q to cancel", dim=True))
        return lines

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        # Hide cursor and render initial output
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        lines = render()
        line_count = len(lines)
        click.echo("\n".join(lines))

        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # Escape sequence
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A":  # Up arrow
                            selected_idx = (selected_idx - 1) % len(organizations)
                        elif ch3 == "B":  # Down arrow
                            selected_idx = (selected_idx + 1) % len(organizations)
                    elif ch2 == "\x1b" or ch2 == "":  # Double escape or timeout
                        # Restore terminal and clean up
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        sys.stdout.write("\033[?25h\n")
                        sys.stdout.flush()
                        return None
                elif ch == "k":  # Vim up
                    selected_idx = (selected_idx - 1) % len(organizations)
                elif ch == "j":  # Vim down
                    selected_idx = (selected_idx + 1) % len(organizations)
                elif ch == "\r" or ch == "\n":  # Enter
                    # Restore terminal before returning
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write("\033[?25h\n")
                    sys.stdout.flush()
                    return organizations[selected_idx].id
                elif ch == "q" or ch == "\x03":  # q or Ctrl+C
                    # Restore terminal and clean up
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write("\033[?25h\n")
                    sys.stdout.flush()
                    return None

                # Move cursor up to beginning of our output and clear
                # Need to exit raw mode temporarily for proper output
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                sys.stdout.write(f"\033[{line_count}A")  # Move up
                sys.stdout.write("\033[J")  # Clear from cursor to end of screen
                lines = render()
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
                tty.setraw(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h")  # Show cursor
            sys.stdout.flush()
    except (ImportError, termios.error):
        # Fallback for non-Unix systems
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.flush()
        return None


@click.command()
@click.argument("org_id", required=False)
@click.pass_context
def orgs(ctx, org_id: str | None):
    """Select active organization interactively or by ID.

    Without arguments, shows interactive selector.
    With ORG_ID, sets that organization as active directly.
    """
    require_auth()
    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")

    # Fetch organizations first (needed for both direct ID and interactive)
    try:
        response = api_request("get", "/auth/organizations", base_url)
        response.raise_for_status()
        orgs_data = models.Organizations(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid organization data: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to fetch organizations: {e}")

    # If org_id provided, validate and select it directly
    if org_id:
        # Check if org_id exists in user's organizations
        valid_org = next((org for org in orgs_data.organizations if org.id == org_id), None)
        if not valid_org:
            click.echo(click.style(f"✗ Organization not found: {org_id}", fg="red"))
            click.echo()
            click.echo("Available organizations:")
            for org in orgs_data.organizations:
                click.echo(f"  • {org.id} ({org.name})")
            raise click.ClickException("Invalid organization ID")

        auth.set_active_organization(org_id)
        click.echo(click.style(f"✓ Active organization set to: {valid_org.name}", fg=TEAL_RGB))
        return

    if not orgs_data.organizations:
        click.echo(click.style("No organizations found", fg="yellow"))
        return

    # Get locally stored active org
    active_org_id = get_active_organization()

    # Show interactive selector
    selected_org_id = _interactive_org_selector(orgs_data.organizations, active_org_id)

    if selected_org_id and selected_org_id != active_org_id:
        auth.set_active_organization(selected_org_id)
        # Find org name for display
        org_name = next(
            (org.name for org in orgs_data.organizations if org.id == selected_org_id),
            selected_org_id,
        )
        click.echo()
        click.echo(click.style(f"✓ Switched to: {org_name}", fg=TEAL_RGB))
    elif selected_org_id:
        click.echo()
        click.echo(click.style("Organization unchanged", dim=True))


# ========== Project Commands ==========


@click.command()
@click.option(
    "--template",
    "-t",
    type=click.Choice(
        [
            "start",
            "new",
            "blank",
            "sft",
            "rl-single",
            "rl-nextjs",
            "rl-tools",
            "mcp-tavily",
            "deepseek-aha",
            "tutorial-reward",
            "tutorial-tool",
        ]
    ),
    default="start",
    help="Project template to use",
)
@click.option("--name", "-n", help="Project name (will prompt if not provided)")
def init(template: str, name: str):
    """Initialize a new ReinforceNow project."""
    require_auth()

    import shutil
    from pathlib import Path

    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import HTML

    def styled_prompt(question: str, default: str) -> str:
        """Next.js style prompt with placeholder that disappears on typing."""
        result = pt_prompt(
            HTML(f"<b>{question}</b> <gray>›</gray> "),
            placeholder=HTML(f"<gray>{default}</gray>"),
        )
        return result.strip() or default

    # Map "start" to "rl-single"
    actual_template = "rl-single" if template == "start" else template

    # Default project names based on template
    template_default_names = {
        "rl-single": "rl-project",
        "rl-tools": "rl-tools-project",
        "rl-nextjs": "nextjs-project",
        "mcp-tavily": "mcp-tavily-project",
        "sft": "sft-project",
        "tutorial-reward": "tutorial-reward",
        "tutorial-tool": "tutorial-tool",
        "deepseek-aha": "deepseek-aha",
        "new": "new-project",
        "blank": "my-project",
    }
    default_project_name = template_default_names.get(actual_template, "my-project")

    # Project name prompt
    project_name = (
        name if name else styled_prompt("What is your project named?", default_project_name)
    )

    # Dataset name prompt
    dataset_name = styled_prompt("What is your dataset named?", "train")

    # Create project directory in current location
    project_dir = Path(".")

    # Copy template files if template is specified (all except blank)
    if actual_template != "blank":
        template_dir = Path(__file__).parent.parent / "templates" / actual_template
        if template_dir.exists():
            # Get list of files to copy from template
            files_to_copy = [f for f in template_dir.iterdir() if f.is_file()]
            template_file_names = {f.name for f in files_to_copy}

            # Define template-managed files (files that templates can provide)
            managed_files = {
                "config.yml",
                "train.jsonl",
                "rewards.py",
                "requirements.txt",
                "env.py",
                "README.md",
            }

            # Find template-managed files that exist but aren't in the new template
            extra_files = [
                fname
                for fname in managed_files
                if (project_dir / fname).exists() and fname not in template_file_names
            ]

            # Check if any files will be overwritten
            existing_files = [f.name for f in files_to_copy if (project_dir / f.name).exists()]

            # Show concise warning and confirm
            if extra_files or existing_files:
                all_affected = extra_files + existing_files
                click.echo(
                    click.style("Files to modify:", bold=True)
                    + click.style(f" {', '.join(all_affected)}", dim=True)
                )
                confirm_prompt = (
                    click.style("Continue?", bold=True)
                    + " ("
                    + click.style("yes", dim=True)
                    + "/no)"
                )
                if not click.confirm(
                    confirm_prompt, default=True, show_default=False, prompt_suffix=" "
                ):
                    raise click.Abort()

            # Remove extra template files (silently)
            for fname in extra_files:
                (project_dir / fname).unlink()

            # Copy all template files to current directory (silently)
            for file in files_to_copy:
                dest_file = project_dir / file.name
                shutil.copy2(file, dest_file)
        else:
            click.echo(
                click.style("Template not found:", bold=True)
                + click.style(f" {template}, using blank template", dim=True)
            )

    # Generate new IDs
    project_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())
    org_id = get_active_organization()

    # Update config.yml with actual IDs
    config_path = project_dir / "config.yml"
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Update IDs and name
        config_data["project_id"] = project_id
        config_data["project_name"] = project_name
        config_data["dataset_id"] = dataset_id
        config_data["dataset_name"] = dataset_name
        config_data["organization_id"] = org_id

        # Reorder keys to ensure proper field ordering in output
        key_order = [
            "project_id",
            "project_name",
            "dataset_id",
            "dataset_name",
            "dataset_type",
            "organization_id",
            "data",
            "model",
            "algorithm",
            "rollout",
            "trainer",
        ]
        ordered_config = {k: config_data[k] for k in key_order if k in config_data}
        # Add any remaining keys not in the order list
        for k in config_data:
            if k not in ordered_config:
                ordered_config[k] = config_data[k]

        with open(config_path, "w") as f:
            yaml.dump(ordered_config, f, default_flow_style=False, sort_keys=False)
    else:
        # Create new config for blank template
        config = models.ProjectConfig(
            project_id=project_id,
            project_name=project_name,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_type=models.DatasetType.RL,
            organization_id=org_id,
            data=models.DataConfig(batch_size=2, group_size=16),
            model=models.ModelConfig(path="Qwen/Qwen3-8B"),
            trainer=models.TrainerConfig(num_epochs=30),
        )

        with open(config_path, "w") as f:
            yaml.dump(
                config.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        pass  # Config created silently

    click.echo(click.style(f"\n✓ Created local project: {project_name}", fg=TEAL_RGB))
    click.echo()
    click.echo(click.style("Next steps:", bold=True))
    click.echo(f"  1. Edit {click.style('train.jsonl', underline=True)} with your training data")
    click.echo(
        f"  2. Edit {click.style('rewards.py', underline=True)} and {click.style('env.py', underline=True)} with your reward and tool functions"
    )
    click.echo(f"  3. Run {click.style('rnow run', fg=TEAL_RGB, bold=True)} to start training")


def parse_override(override: str) -> tuple[list[str], any]:
    """
    Parse a single override string like 'algorithm.adv_estimator=grpo'.

    Returns:
        Tuple of (key_path, value) where key_path is a list of nested keys.
    """
    if "=" not in override:
        raise click.ClickException(
            f"Invalid override '{override}'. Use format: key=value or nested.key=value"
        )

    key, value = override.split("=", 1)
    key_path = key.strip().split(".")

    # Try to parse value as JSON (for numbers, bools, lists)
    value = value.strip()
    if value.lower() == "true":
        return key_path, True
    elif value.lower() == "false":
        return key_path, False
    elif value.lower() == "null" or value.lower() == "none":
        return key_path, None

    # Try numeric
    try:
        if "." in value:
            return key_path, float(value)
        else:
            return key_path, int(value)
    except ValueError:
        pass

    # Keep as string
    return key_path, value


def apply_overrides(config_data: dict, overrides: tuple[str, ...]) -> dict:
    """
    Apply CLI overrides to config data.

    Args:
        config_data: The loaded config dictionary
        overrides: Tuple of override strings like ('algorithm.adv_estimator=grpo', 'model.path=Qwen/Qwen3-4B')

    Returns:
        Modified config data
    """
    for override in overrides:
        key_path, value = parse_override(override)

        # Navigate to the nested location
        current = config_data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise click.ClickException(
                    f"Cannot override '{'.'.join(key_path)}': '{key}' is not a nested object"
                )
            current = current[key]

        # Set the value
        final_key = key_path[-1]
        old_value = current.get(final_key, "<not set>")
        current[final_key] = value

        # Show what was changed
        full_key = ".".join(key_path)
        click.echo(f"  Override: {click.style(full_key, fg=TEAL_RGB)} = {value} (was: {old_value})")

    return config_data


def _submit_single_run(
    ctx,
    dir: Path,
    config_data: dict,
    base_url: str,
    name: str | None,
    debug: bool,
    model_override: str | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    overrides: tuple[str, ...],
) -> dict | None:
    """
    Submit a single training run. Returns dict with run_id and run_url on success.
    Raises click.ClickException on failure.
    """
    # Build combined overrides from shorthand options + explicit overrides
    all_overrides = list(overrides)

    # Add shorthand options as overrides (skip model override - already set in config_data)
    if epochs:
        all_overrides.insert(0, f"trainer.num_epochs={epochs}")
    if batch_size:
        all_overrides.insert(0, f"data.batch_size={batch_size}")
    if lr:
        all_overrides.insert(0, f"trainer.learning_rate={lr}")

    # Apply CLI overrides before validation
    if all_overrides:
        config_data = apply_overrides(config_data, tuple(all_overrides))

    # Now validate the config with overrides applied
    try:
        config = models.ProjectConfig(**config_data)
    except ValidationError as e:
        raise click.ClickException(format_validation_error(e))

    if not config.organization_id:
        config.organization_id = get_active_organization()

    # Validate required files
    required_files = {"train.jsonl": dir / "train.jsonl"}
    if config.dataset_type == models.DatasetType.RL:
        required_files["rewards.py"] = dir / "rewards.py"

    for file_name, path in required_files.items():
        if not path.exists():
            raise click.ClickException(f"Missing required file: {file_name}")
        elif path.stat().st_size == 0:
            raise click.ClickException(f"Empty file: {file_name}")

    # Validate train.jsonl format
    train_jsonl_path = dir / "train.jsonl"
    if train_jsonl_path.exists() and train_jsonl_path.stat().st_size > 0:
        jsonl_errors = validate_train_jsonl(train_jsonl_path, config.dataset_type)
        if jsonl_errors:
            raise click.ClickException(f"Invalid train.jsonl: {jsonl_errors[0]}")

    # Validate rewards.py if present
    if config.dataset_type == models.DatasetType.RL:
        rewards_path = dir / "rewards.py"
        if rewards_path.exists():
            try:
                from rnow.core.reward import validate_rewards_file

                errors = validate_rewards_file(rewards_path)
                if errors:
                    raise click.ClickException(f"Invalid rewards.py: {errors[0]}")
            except ImportError:
                pass

            ref_errors = validate_reward_references(train_jsonl_path, rewards_path)
            if ref_errors:
                raise click.ClickException(f"Reward mismatch: {ref_errors[0]}")

        # Validate context window (prompt + tools + max_tokens)
        if config.rollout:
            model_path = config.model.path if config.model else ""
            model_name = model_path.split("/")[-1] if "/" in model_path else model_path

            click.echo(f"  [{model_name}] Validating context window...")

            # Collect all tools
            all_tools = []

            # Get tools from env.py
            env_path = dir / "env.py"
            env_tools = get_tools_from_env_py(env_path)
            all_tools.extend(env_tools)

            # Fetch MCP tools
            mcp_urls = config.rollout.mcp_url
            if mcp_urls:
                mcp_tools, mcp_error = fetch_mcp_tool_schemas(mcp_urls, timeout=15.0)
                if mcp_error:
                    raise click.ClickException(
                        f"Failed to fetch MCP tools for {model_name}: {mcp_error}"
                    )
                all_tools.extend(mcp_tools)
                click.echo(f"  [{model_name}] MCP tools: {len(mcp_tools)} tools")

            # Count tokens with proper format (includes Harmony rendering for gpt-oss)
            total_prompt_tokens = get_max_prompt_tokens(train_jsonl_path, all_tools, model_path)

            click.echo(
                f"  [{model_name}] Total: {total_prompt_tokens:,} + {config.rollout.max_tokens:,} = {total_prompt_tokens + config.rollout.max_tokens:,} / {models.MAX_CONTEXT_WINDOW:,}"
            )

            context_error, recommended = validate_max_tokens_for_context(
                config.rollout.max_tokens, total_prompt_tokens
            )
            if context_error:
                raise click.ClickException(
                    f"Context window exceeded for {model_name}: "
                    f"~{total_prompt_tokens:,} prompt+tools + {config.rollout.max_tokens:,} max_tokens "
                    f"> {models.MAX_CONTEXT_WINDOW:,}. Set max_tokens to {recommended:,} or less."
                )

    # Check if train.jsonl needs blob upload
    train_path = dir / "train.jsonl"
    train_size = train_path.stat().st_size
    dataset_url = None

    if train_size > MAX_INLINE_BYTES:
        try:
            _, blob_info = maybe_upload_to_blob(base_url, train_path, config.dataset_id)
            if blob_info:
                dataset_url = blob_info.get("url")
        except Exception as e:
            raise click.ClickException(f"Failed to upload large dataset: {e}")

    # Upload files
    files = []

    # Add config file - create a temporary one with the modified config
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml.dump(config_data, tmp, default_flow_style=False, sort_keys=False)
        tmp_config_path = Path(tmp.name)

    files.append(
        ("config_yml", ("config.yml", open(tmp_config_path, "rb"), "application/octet-stream"))
    )

    # Add required files (skip train.jsonl if uploaded to blob)
    for file_name, path in required_files.items():
        if file_name == "train.jsonl" and dataset_url:
            continue
        files.append(
            (file_name.replace(".", "_"), (file_name, open(path, "rb"), "application/octet-stream"))
        )

    # Add optional files
    optional_files = {"env.py": dir / "env.py", "requirements.txt": dir / "requirements.txt"}
    for file_name, path in optional_files.items():
        if path.exists():
            files.append(
                (
                    file_name.replace(".", "_"),
                    (file_name, open(path, "rb"), "application/octet-stream"),
                )
            )

    headers = auth.get_auth_headers()
    headers.pop("Content-Type", None)

    submit_data = {
        "project_id": config.project_id,
        "dataset_id": config.dataset_id,
        "organization_id": config.organization_id,
    }
    if name:
        submit_data["run_name"] = name
    if dataset_url:
        submit_data["dataset_url"] = dataset_url
    if debug:
        submit_data["debug"] = "true"

    run_url = None
    run_id = None
    error_msg = None

    try:
        response = session.post(
            f"{base_url}/training/submit",
            data=submit_data,
            files=files,
            headers=headers,
            stream=True,
        )

        if response.status_code != 200:
            error_msg = f"Training submission failed: {response.text}"
        else:
            response.encoding = "utf-8"
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    msg = line[6:]
                    if "View:" in msg:
                        run_url = msg.split("View:")[-1].strip()
                        if run_url:
                            run_id = run_url.rstrip("/").split("/")[-1]
                    elif "http" in msg and "View" not in msg:
                        run_url = msg.split()[-1].strip()
                        if run_url:
                            run_id = run_url.rstrip("/").split("/")[-1]
                    elif msg.startswith("❌") or "Error" in msg or "failed" in msg.lower():
                        error_msg = msg

    except Exception as e:
        error_msg = f"Request failed: {e}"
    finally:
        for _, (_, fh, _) in files:
            fh.close()
        # Clean up temp config file
        tmp_config_path.unlink(missing_ok=True)

    if error_msg:
        raise click.ClickException(error_msg)

    return {"run_id": run_id, "run_url": run_url}


@click.command()
@click.option(
    "--dir",
    "-d",
    default=".",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing project files (default: current directory)",
)
@click.option(
    "--name", "-n", default=None, help="Custom name for the training run (default: auto-generated)"
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Debug mode: upload files but don't start training job",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Override model path (e.g., Qwen/Qwen3-4B)",
)
@click.option(
    "--epochs",
    "-e",
    default=None,
    type=int,
    help="Override number of training epochs",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help="Override batch size (1-32)",
)
@click.option(
    "--lr",
    "--learning-rate",
    default=None,
    type=float,
    help="Override learning rate",
)
@click.argument("overrides", nargs=-1)
@click.pass_context
def run(
    ctx,
    dir: Path,
    name: str,
    debug: bool,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    overrides: tuple[str, ...],
):
    """Submit project for training on ReinforceNow platform.

    You can override any config.yml setting by passing key=value arguments:

    \b
    Examples:
        rnow run model.path=Qwen/Qwen3-4B
        rnow run algorithm.adv_estimator=grpo trainer.learning_rate=0.0002
        rnow run data.batch_size=8 data.group_size=16 trainer.num_epochs=5
        rnow run rollout.max_turns=3 rollout.max_tokens=4096

    \b
    Common overrides:
        model.path              Model to train (e.g., Qwen/Qwen3-8B, Qwen/Qwen3-4B)
        model.qlora_rank        LoRA rank (default: 32)
        data.batch_size         Batch size (1-32)
        data.group_size         Rollouts per prompt for RL (1-64)
        trainer.num_epochs      Number of training epochs
        trainer.learning_rate   Learning rate (default: 0.0001)
        algorithm.adv_estimator Advantage estimator: grpo, gae, reinforce
        algorithm.loss_fn       Loss function: ppo, importance_sampling
        rollout.max_turns       Max conversation turns for RL
        rollout.max_tokens      Max tokens per generation
        rollout.thinking_mode   Reasoning mode: disabled, easy, medium, hard

    Multi-model training:
        If model.path is a list in config.yml, a separate run will be submitted
        for each model in the list.

        Example config.yml:
            model:
              path:
                - Qwen/Qwen3-8B
                - Qwen/Qwen3-4B
                - meta-llama/Llama-3.1-8B-Instruct
    """
    require_auth()
    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")

    # Load and validate config
    config_yml = dir / "config.yml"
    config_json = dir / "config.json"

    # First load raw config data
    config_data = None
    if config_yml.exists():
        try:
            with open(config_yml) as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise click.ClickException(f"Config file not found in {dir}")
        except yaml.YAMLError as e:
            raise click.ClickException(f"Invalid YAML in config file: {e}")
    elif config_json.exists():
        try:
            with open(config_json) as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in config file: {e}")
    else:
        raise click.ClickException(f"No config.yml or config.json found in {dir}")

    # Check if model.path is a list (multi-model training)
    model_paths = config_data.get("model", {}).get("path")
    if isinstance(model_paths, list) and len(model_paths) > 1:
        # Validate qlora_rank for each model before starting any runs
        qlora_rank = config_data.get("model", {}).get("qlora_rank", 32)
        for model_path in model_paths:
            max_rank = models.get_max_lora_rank(model_path)
            if qlora_rank > max_rank:
                model_name = model_path.split("/")[-1] if "/" in model_path else model_path
                raise click.ClickException(
                    f"qlora_rank {qlora_rank} exceeds maximum {max_rank} for model {model_name}. "
                    f"Set qlora_rank to {max_rank} or lower to train all models."
                )

        results = []
        for model_path in model_paths:
            model_name = model_path.split("/")[-1] if "/" in model_path else model_path

            # Create a copy of config_data with single model path
            single_config = json.loads(json.dumps(config_data))  # Deep copy
            single_config["model"]["path"] = model_path

            # Submit this model's run
            try:
                run_result = _submit_single_run(
                    ctx=ctx,
                    dir=dir,
                    config_data=single_config,
                    base_url=base_url,
                    name=name,
                    debug=debug,
                    model_override=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    overrides=overrides,
                )
                results.append((model_path, run_result, None))
            except click.ClickException as e:
                results.append((model_path, None, str(e)))

        # Show run IDs
        click.echo(click.style("Run IDs:", bold=True))
        for model_path, result, _error in results:
            model_name = model_path.split("/")[-1] if "/" in model_path else model_path
            if result and result.get("run_id"):
                click.echo(f"  {model_name}: {result['run_id']}")
            else:
                click.echo(f"  {model_name}: {click.style('failed', fg='red')}")

        # Show run URLs
        successful = [r for r in results if r[1] is not None]
        if successful:
            click.echo()
            click.echo(click.style("Run URLs:", bold=True))
            for model_path, result, _ in successful:
                model_name = model_path.split("/")[-1] if "/" in model_path else model_path
                if result and result.get("run_url"):
                    click.echo(f"  {model_name}: {click.style(result['run_url'], fg=TEAL_RGB)}")

        return

    # Single model training - continue with normal flow
    # Build combined overrides from shorthand options + explicit overrides
    all_overrides = list(overrides)

    # Add shorthand options as overrides
    if model:
        all_overrides.insert(0, f"model.path={model}")
    if epochs:
        all_overrides.insert(0, f"trainer.num_epochs={epochs}")
    if batch_size:
        all_overrides.insert(0, f"data.batch_size={batch_size}")
    if lr:
        all_overrides.insert(0, f"trainer.learning_rate={lr}")

    # Apply CLI overrides before validation
    if all_overrides:
        click.echo(click.style("Applying config overrides:", bold=True))
        config_data = apply_overrides(config_data, tuple(all_overrides))
        click.echo()

    # Now validate the config with overrides applied
    try:
        config = models.ProjectConfig(**config_data)
    except ValidationError as e:
        click.echo(format_validation_error(e))
        if overrides:
            click.echo(
                click.style("\nHint: One of your overrides may have an invalid value.", fg="yellow")
            )
        raise click.ClickException("Please fix config before submitting")

    if not config.organization_id:
        config.organization_id = get_active_organization()

    # Validate required files (all in the same directory now)
    required_files = {
        "train.jsonl": dir / "train.jsonl",
    }

    # Only require rewards.py for RL datasets
    if config.dataset_type == models.DatasetType.RL:
        required_files["rewards.py"] = dir / "rewards.py"

    missing_files = []
    empty_files = []
    for file_name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"  • {file_name} at {path}")
        elif path.stat().st_size == 0:
            if file_name == "train.jsonl":
                empty_files.append(
                    f"  • {file_name} is empty - please add training examples (one JSON object per line)"
                )
            elif file_name == "rewards.py":
                empty_files.append(
                    f"  • {file_name} is empty - please implement your reward function"
                )

    if missing_files:
        click.echo(click.style("✗ Required files missing:", fg="red", bold=True))
        for file_msg in missing_files:
            click.echo(file_msg)
        raise click.ClickException("Missing required files for training submission")

    if empty_files:
        click.echo(click.style("✗ Empty files detected:", fg="red", bold=True))
        for file_msg in empty_files:
            click.echo(file_msg)
        raise click.ClickException("Please add content to empty files before submitting")

    # Validate train.jsonl format (sample first 50 lines)
    train_jsonl_path = dir / "train.jsonl"
    if train_jsonl_path.exists() and train_jsonl_path.stat().st_size > 0:
        jsonl_errors = validate_train_jsonl(train_jsonl_path, config.dataset_type)
        if jsonl_errors:
            click.echo(click.style("✗ Invalid train.jsonl format:", fg="red", bold=True))
            for err in jsonl_errors:
                click.echo(f"  • {err}")
            raise click.ClickException("Please fix train.jsonl format before submitting")

        # Validate max_tokens vs prompt size for RL (including tool definitions)
        if config.dataset_type == models.DatasetType.RL and config.rollout:
            # Get model path for accurate tokenization
            model_path = config.model.path if config.model else ""

            # Try to load tokenizer (show message if loading)
            if model_path:
                click.echo(click.style("Loading tokenizer...", dim=True), nl=False)
                tokenizer_info = get_tokenizer_for_model(model_path)
                if tokenizer_info:
                    tokenizer_type = tokenizer_info[0]
                    label = "Harmony" if tokenizer_type == "harmony" else "HuggingFace"
                    click.echo(
                        "\r"
                        + click.style("Tokenizer: ", fg=TEAL_RGB)
                        + f"{label} ({model_path})"
                        + " " * 10
                    )
                else:
                    click.echo(
                        "\r"
                        + click.style("Tokenizer: ", fg="yellow")
                        + "not available, using estimates"
                        + " " * 10
                    )

            # Collect all tools
            all_tools = []

            # Get tools from env.py
            env_path = dir / "env.py"
            env_tools = get_tools_from_env_py(env_path)
            all_tools.extend(env_tools)

            # Fetch MCP tool schemas (with progress indicator)
            mcp_urls = config.rollout.mcp_url if config.rollout else None
            if mcp_urls:
                # Check if fastmcp is installed
                try:
                    import fastmcp  # noqa: F401
                except ImportError:
                    click.echo()
                    click.echo(click.style("✗ MCP support requires fastmcp", fg="red", bold=True))
                    click.echo()
                    click.echo("  Your config.yml uses mcp_url, but fastmcp is not installed")
                    click.echo("  in the same Python environment as rnow.")
                    click.echo()
                    click.echo(f"  rnow is running from: {click.style(sys.executable, dim=True)}")
                    click.echo()
                    click.echo("  Install it with:")
                    click.echo(
                        click.style(f"    {sys.executable} -m pip install fastmcp", fg=TEAL_RGB)
                    )
                    click.echo()
                    raise click.ClickException("Missing dependency: fastmcp")

                click.echo(click.style("Fetching MCP tools...", dim=True), nl=False)
                mcp_tools, mcp_error = fetch_mcp_tool_schemas(mcp_urls, timeout=15.0)
                if mcp_error:
                    click.echo(
                        "\r" + click.style("MCP: ", fg="red") + f"failed ({mcp_error})" + " " * 20
                    )
                    raise click.ClickException(f"Failed to fetch MCP tools: {mcp_error}")

                all_tools.extend(mcp_tools)
                click.echo(
                    "\r" + click.style("MCP: ", fg=TEAL_RGB) + f"{len(mcp_tools)} tools" + " " * 20
                )

            # Count tokens with proper format (includes Harmony rendering for gpt-oss)
            total_prompt_tokens = get_max_prompt_tokens(train_jsonl_path, all_tools, model_path)

            # Show context window usage
            if total_prompt_tokens > 0:
                is_gpt_oss = "gpt-oss" in model_path.lower()
                format_note = " (Harmony format)" if is_gpt_oss else ""
                click.echo(
                    click.style("Context: ", fg=TEAL_RGB)
                    + f"~{total_prompt_tokens:,} prompt+tools{format_note}"
                    + f" + {config.rollout.max_tokens:,} max_tokens"
                    + f" = ~{total_prompt_tokens + config.rollout.max_tokens:,}"
                    + f" / {models.MAX_CONTEXT_WINDOW:,}"
                )
                context_error, recommended = validate_max_tokens_for_context(
                    config.rollout.max_tokens, total_prompt_tokens
                )
                if context_error:
                    click.echo()
                    click.echo(click.style("✗ Context window exceeded", fg="red", bold=True))
                    click.echo()
                    click.echo(
                        f"  Total prompt context (with tools): ~{total_prompt_tokens:,} tokens."
                    )
                    if is_gpt_oss:
                        click.echo(
                            "  Note: gpt-oss uses Harmony format which includes system overhead."
                        )
                    click.echo(
                        f"  With max_tokens={config.rollout.max_tokens:,}, the total exceeds"
                    )
                    click.echo(f"  the {models.MAX_CONTEXT_WINDOW:,} token context window.")
                    click.echo()
                    click.echo(
                        click.style("  Fix:", bold=True)
                        + f" Set rollout.max_tokens to {recommended:,} or less"
                    )
                    click.echo()
                    click.echo(click.style("  In config.yml:", dim=True))
                    click.echo(click.style("    rollout:", dim=True))
                    click.echo(f"      max_tokens: {click.style(str(recommended), fg=TEAL_RGB)}")
                    click.echo()
                    raise click.ClickException("max_tokens + prompt length exceeds context window")

    # Validate requirements.txt if present (check format and Python 3.11 compatibility)
    requirements_path = dir / "requirements.txt"
    if requirements_path.exists() and requirements_path.stat().st_size > 0:
        req_errors = validate_requirements_txt(requirements_path, target_python="3.11")
        if req_errors:
            click.echo(click.style("✗ Invalid requirements.txt:", fg="red", bold=True))
            for err in req_errors:
                click.echo(f"  • {err}")
            raise click.ClickException("Please fix requirements.txt before submitting")

    # Validate rewards.py if present (check signature on @reward functions)
    if config.dataset_type == models.DatasetType.RL:
        rewards_path = dir / "rewards.py"
        if rewards_path.exists():
            try:
                from rnow.core.reward import validate_rewards_file

                errors = validate_rewards_file(rewards_path)
                if errors:
                    click.echo(click.style("✗ Invalid rewards.py:", fg="red", bold=True))
                    for err in errors:
                        click.echo(f"  • {err}")
                    raise click.ClickException("Please fix rewards.py before submitting")
            except ImportError:
                pass  # Skip validation if module not available

            # Validate that rewards referenced in train.jsonl exist in rewards.py
            ref_errors = validate_reward_references(train_jsonl_path, rewards_path)
            if ref_errors:
                click.echo(click.style("✗ Reward mismatch:", fg="red", bold=True))
                for err in ref_errors:
                    click.echo(f"  • {err}")
                raise click.ClickException(
                    "Please ensure reward names in train.jsonl match functions in rewards.py"
                )

    # Validate env.py if present (check for docstrings on @tool functions)
    env_path = dir / "env.py"
    has_env_py = env_path.exists() and env_path.stat().st_size > 0
    if has_env_py:
        try:
            from rnow.core.tool import validate_tools_file

            errors = validate_tools_file(env_path)
            if errors:
                click.echo(click.style("✗ Invalid env.py:", fg="red", bold=True))
                for err in errors:
                    click.echo(f"  • {err}")
                raise click.ClickException("Please fix env.py before submitting")
        except ImportError:
            pass  # Skip validation if module not available

    # Check for MCP URL(s) in config
    has_mcp_url = config.rollout is not None and config.rollout.mcp_url is not None
    mcp_url_count = 0
    if has_mcp_url:
        mcp_url = config.rollout.mcp_url
        mcp_url_count = len(mcp_url) if isinstance(mcp_url, list) else 1

    # Show tool sources message
    if has_env_py and has_mcp_url:
        server_text = f"{mcp_url_count} server(s)" if mcp_url_count > 1 else "1 server"
        click.echo(
            click.style("Tools: ", fg=TEAL_RGB) + f"Using MCP ({server_text}) and env.py tools"
        )
    elif has_mcp_url:
        server_text = f"{mcp_url_count} server(s)" if mcp_url_count > 1 else "1 server"
        click.echo(click.style("Tools: ", fg=TEAL_RGB) + f"Using MCP ({server_text})")
    elif has_env_py:
        click.echo(click.style("Tools: ", fg=TEAL_RGB) + "Using env.py tools")

    # Start cube spinner early
    spinner = CubeSpinner()

    # Check if train.jsonl needs blob upload (>4MB)
    train_path = dir / "train.jsonl"
    train_size = train_path.stat().st_size
    dataset_url = None

    if train_size > MAX_INLINE_BYTES:
        spinner.start()
        try:
            _, blob_info = maybe_upload_to_blob(base_url, train_path, config.dataset_id)
            if blob_info:
                dataset_url = blob_info.get("url")
        except Exception as e:
            spinner.stop()
            raise click.ClickException(f"Failed to upload large dataset: {e}")

    # Upload files
    files = []

    # Add config file
    if config_yml.exists():
        files.append(
            ("config_yml", ("config.yml", open(config_yml, "rb"), "application/octet-stream"))
        )
    elif config_json.exists():
        files.append(
            ("config_json", ("config.json", open(config_json, "rb"), "application/octet-stream"))
        )

    # Add required files (skip train.jsonl if uploaded to blob)
    for file_name, path in required_files.items():
        if file_name == "train.jsonl" and dataset_url:
            # Skip - already uploaded to blob
            continue
        files.append(
            (file_name.replace(".", "_"), (file_name, open(path, "rb"), "application/octet-stream"))
        )

    # Add optional files (all in the same directory now)
    optional_files = {
        "env.py": dir / "env.py",
        "requirements.txt": dir / "requirements.txt",
    }

    for file_name, path in optional_files.items():
        if path.exists():
            files.append(
                (
                    file_name.replace(".", "_"),
                    (file_name, open(path, "rb"), "application/octet-stream"),
                )
            )

    # For multipart, we need to omit Content-Type so requests sets the boundary
    headers = auth.get_auth_headers()
    headers.pop("Content-Type", None)

    # Include custom run name if provided
    submit_data = {
        "project_id": config.project_id,
        "dataset_id": config.dataset_id,
        "organization_id": config.organization_id,
    }
    if name:
        submit_data["run_name"] = name

    # Add dataset URL if uploaded to blob
    if dataset_url:
        submit_data["dataset_url"] = dataset_url

    # Add debug flag if set
    if debug:
        submit_data["debug"] = "true"

    # Start cube spinner if not already running (for small files)
    if not spinner.running:
        spinner.start()

    run_url = None
    run_id = None
    error_msg = None
    resolved_base_model = None
    resolved_finetuned_model = None

    try:
        response = session.post(
            f"{base_url}/training/submit",
            data=submit_data,
            files=files,
            headers=headers,
            stream=True,
        )

        if response.status_code != 200:
            error_msg = f"Training submission failed: {response.text}"
        else:
            response.encoding = "utf-8"

            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    msg = line[6:]

                    if "View:" in msg:
                        run_url = msg.split("View:")[-1].strip()
                        # Extract run_id from URL (last path segment)
                        if run_url:
                            run_id = run_url.rstrip("/").split("/")[-1]
                    elif "http" in msg and "View" not in msg:
                        run_url = msg.split()[-1].strip()
                        if run_url:
                            run_id = run_url.rstrip("/").split("/")[-1]
                    elif msg.startswith("❌") or "Error" in msg or "failed" in msg.lower():
                        error_msg = msg
                    # Capture resolved model info from server
                    elif "Resuming from finetuned model:" in msg:
                        resolved_finetuned_model = msg.split("Resuming from finetuned model:")[
                            -1
                        ].strip()
                    elif "Base model:" in msg:
                        resolved_base_model = msg.split("Base model:")[-1].strip()

    except Exception as e:
        error_msg = f"Request failed: {e}"
    finally:
        for _, (_, fh, _) in files:
            fh.close()

    # Show result
    if error_msg:
        spinner.stop()  # Clear cube on error
        click.echo(click.style(f"✗ {error_msg}", fg="red", bold=True))
        raise click.ClickException("Training submission failed")

    # Stop spinner but keep cube visible on success
    spinner.stop(keep_visible=True)

    # Get display values
    model_path = config.model.path if config.model else "Qwen/Qwen3-8B"
    thinking_mode = get_thinking_mode_display(config)

    # Build model display string
    thinking_styled = click.style(thinking_mode, fg=TEAL_RGB)
    if resolved_finetuned_model and resolved_base_model:
        # Resuming from a finetuned model - show both
        model_display = f"{resolved_finetuned_model} ({thinking_styled})"
        base_model_display = f"  Base: {resolved_base_model}"
    else:
        # Fresh training from base model
        model_display = f"{model_path} ({thinking_styled})"
        base_model_display = None

    # Output completion messages below the cube
    click.echo(f"Run started successfully {click.style('✅', fg=TEAL_RGB)}")
    click.echo(f"  Project: {config.project_name}")
    click.echo(f"  Model: {model_display}")
    if base_model_display:
        click.echo(base_model_display)
    if run_id:
        click.echo(f"  Run ID: {run_id}")
    if run_url:
        click.echo("\nView your experiment here:")
        click.echo(click.style(run_url, fg=TEAL_RGB))


@click.command()
@click.argument("model_id", required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for extracted checkpoint (default: ./<model_name>/)",
)
@click.option(
    "--keep-archive", is_flag=True, default=False, help="Keep the tar archive after extraction"
)
@click.pass_context
def download(ctx, model_id: str, output: Path, keep_archive: bool):
    """Download a trained model by ID.

    Downloads and extracts the model checkpoint (LoRA adapter weights).
    The checkpoint is downloaded as a tar archive and automatically extracted.

    Examples:

        # Download and extract to ./My_Model/
        rnow download abc123

        # Download and extract to ./models/
        rnow download abc123 --output ./models/

        # Keep the tar archive after extraction
        rnow download abc123 --keep-archive
    """
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")

    # Get download URL from API
    click.echo(f"Fetching download URL for model: {model_id}...")

    try:
        response = api_request("get", f"/models/{model_id}/download", base_url)

        if response.status_code == 404:
            raise click.ClickException("Model not found or no file available for download")
        elif response.status_code == 403:
            raise click.ClickException(
                "Access denied. You don't have permission to download this model."
            )
        elif response.status_code == 400:
            data = response.json()
            raise click.ClickException(data.get("message", "Cannot download this model"))

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        raise click.ClickException(f"Failed to get download URL: {e}")

    download_url = data.get("downloadUrl")
    model_name = data.get("modelName", model_id)
    model_size = int(data.get("modelSize", 0))

    if not download_url:
        raise click.ClickException("No download URL returned from server")

    # Determine output directory
    if output is None:
        # Clean model name for directory
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        output = Path(safe_name)

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Download the file
    click.echo(f"Downloading {model_name}...")

    if model_size > 0:
        size_mb = model_size / (1024 * 1024)
        click.echo(f"  Size: {size_mb:.1f} MB")

    # Use a temp file for the archive
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_file:
        archive_path = Path(tmp_file.name)

    try:
        # Download with progress bar
        with urllib.request.urlopen(download_url, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", model_size))

            with (
                click.progressbar(
                    length=total_size if total_size > 0 else None,
                    label="Downloading",
                    show_pos=total_size > 0,
                    show_percent=True,
                    fill_char=click.style("#", fg=(20, 184, 166)),
                ) as bar,
                open(archive_path, "wb") as f,
            ):
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))

        # Extract the archive
        click.echo("Extracting checkpoint...")
        try:
            with tarfile.open(archive_path, "r") as tar:
                members = tar.getmembers()
                with click.progressbar(
                    members,
                    label="Extracting",
                    show_percent=True,
                    show_pos=True,
                    fill_char=click.style("#", fg=(20, 184, 166)),
                ) as bar:
                    for member in bar:
                        tar.extract(member, path=output)
        except tarfile.TarError as e:
            raise click.ClickException(f"Failed to extract archive: {e}")

        # Optionally keep the archive
        if keep_archive:
            final_archive = output / f"{output.name}.tar"
            shutil.move(str(archive_path), str(final_archive))
            click.echo(f"  Archive saved to: {final_archive}")
        else:
            archive_path.unlink(missing_ok=True)

    except urllib.request.URLError as e:
        archive_path.unlink(missing_ok=True)
        raise click.ClickException(f"Download failed: {e}")
    except Exception as e:
        archive_path.unlink(missing_ok=True)
        raise click.ClickException(f"Download failed: {e}")

    click.echo(
        click.style(
            f"\n✓ Model downloaded and extracted to: {output}/", fg=(20, 184, 166), bold=True
        )
    )

    # List extracted files
    files = list(output.iterdir())
    if files:
        click.echo("\nExtracted files:")
        for f in files[:10]:  # Show first 10 files
            click.echo(f"  • {f.name}")
        if len(files) > 10:
            click.echo(f"  ... and {len(files) - 10} more files")


@click.command()
@click.argument("run_id", required=True)
@click.confirmation_option(prompt="Are you sure you want to stop this training run?")
@click.pass_context
def stop(ctx, run_id: str):
    """Stop an active training run.

    Requires the RUN_ID obtained from 'rnow run' command.
    """
    base_url = ctx.obj.get("api_url", "https://www.reinforcenow.ai/api")

    try:
        click.echo(f"Stopping training run: {run_id}...")
        response = api_request("post", "/training/stop", base_url, json={"run_id": run_id})
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to stop training: {e}")

    click.echo(click.style(f"✓ Training run stopped: {run_id}", fg="green"))

    if data.get("duration_minutes"):
        click.echo(f"  Duration: {data['duration_minutes']:.1f} minutes")
    if data.get("charged_amount"):
        click.echo(f"  Charged: ${data['charged_amount']:.2f}")
