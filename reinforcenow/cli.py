# cli.py
# ReinforceNow CLI with non-blocking login by default and clear UX

import base64
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

from reinforcenow.auth import (
    is_authenticated,
    get_auth_headers,
    login_flow,
    validate_token,
    TOKEN_FILE,
    begin_device_login,
    finish_device_login,
)
from reinforcenow.utils import stream_sse_response

# Load .env from current working directory (optional - has defaults)
load_dotenv()

# Configuration with production default (can be overridden via .env)
API_URL = os.getenv("API_URL", "https://api.reinforcenow.ai")


def get_template_dir():
    """Get the path to the bundled templates directory"""
    return Path(__file__).parent / "templates"


def _not_logged_in_exit(open_browser: bool = True) -> None:
    """
    Standardized behavior for commands that require auth:
    - Tell the user they must log in
    - Open the device authorization page (non-blocking)
    - Exit with code 1
    """
    click.echo("\033[1mNot authenticated.\033[0m CLI authorization required.\n")
    if open_browser:
        pending = begin_device_login()
        if pending:
            click.echo("After authorizing, run your command again.\n")
        else:
            click.echo("Could not start login flow. Run: \033[1mreinforceenow login\033[0m")
    else:
        click.echo("Run: \033[1mreinforceenow login\033[0m")
    sys.exit(1)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--no-wait", is_flag=True, help="Don't wait for authorization (non-blocking). Default: wait.")
@click.option("--force", is_flag=True, help="Ignore existing session and start a fresh login.")
def login(no_wait: bool, force: bool):
    """
    Start the device login flow. By default, waits until authorization completes.
    Use --no-wait to return immediately after opening browser.
    """
    try:
        code = login_flow(wait=not no_wait, force=force)
        sys.exit(code)
    except Exception as e:
        click.echo(f"Login failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--finish/--no-finish", default=True, help="Try to finish an in-progress device login (one-shot, non-blocking).")
@click.option("--wait", is_flag=True, help="Finish login and wait until authorization completes.")
def status(finish: bool, wait: bool):
    """Check authentication status and (optionally) finish any in-progress device login."""
    if TOKEN_FILE.exists():
        click.echo("Checking authentication status...")
        valid = validate_token()
        if valid:
            click.echo("\033[1mAuthenticated\033[0m - token valid")

            # Show token file and decoded expiry (best-effort JWT inspection)
            try:
                with open(TOKEN_FILE) as f:
                    data = json.load(f)
                access_token = data.get("access_token", "")
                click.echo(f"Token file: {TOKEN_FILE}")

                if "." in access_token:
                    try:
                        parts = access_token.split(".")
                        if len(parts) >= 2:
                            payload = parts[1] + "=" * (-len(parts[1]) % 4)
                            decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
                            token_data = json.loads(decoded.decode("utf-8"))
                            if "exp" in token_data:
                                exp_timestamp = token_data["exp"]
                                exp_date = datetime.fromtimestamp(exp_timestamp)
                                click.echo(f"Token expires: {exp_date}")
                    except Exception:
                        pass
            except Exception as e:
                click.echo(f"Could not read token details: {e}")
        else:
            click.echo("\033[1mNot authenticated\033[0m - token invalid or expired")
    else:
        click.echo("\033[1mNot authenticated\033[0m - no token found")

    # Optionally try to finish any pending device login
    if finish:
        rc = finish_device_login(wait=wait)
        if rc == 0:
            click.echo("Session ready.")
        else:
            click.echo("If you just approved, re-run with \033[1m--wait\033[0m to complete.")


@cli.command()
def logout():
    """Clear authentication token."""
    if TOKEN_FILE.exists():
        try:
            TOKEN_FILE.unlink()
            click.echo("\033[1mLogged out.\033[0m Token removed.")
        except Exception as e:
            click.echo(f"Could not remove token: {e}")
    else:
        click.echo("Already logged out - no token found.")


@cli.command()
def start():
    """Initialize a new project with template files."""
    project_dir = Path("./project")
    template_dir = get_template_dir()

    # Create project directory
    project_dir.mkdir(exist_ok=True)

    # List of template files to copy
    template_files = [
        "generation.py",
        "reward_function.py",
        "dataset.jsonl",
        "config.json",
        "project.toml",
    ]

    click.echo("Initializing project with template files...")

    success_count = 0
    failed_files = []

    for filename in template_files:
        source = template_dir / filename
        destination = project_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  Created {filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  Error copying {filename}: {e}")
            failed_files.append(filename)

    # Summary
    click.echo(f"\n\033[1mProject initialized.\033[0m")
    click.echo(f"Created: {success_count}/{len(template_files)} files")

    if failed_files:
        click.echo(f"Failed: {', '.join(failed_files)}")
    else:
        click.echo(f"Files available in: ./project/")
        click.echo(f"\nNext steps:")
        click.echo(f"  1. Edit files in ./project/")
        click.echo(f"  2. Run \033[1mreinforceenow run\033[0m")


def _ensure_auth_or_launch_login() -> None:
    """
    For commands that require auth: if not authenticated,
    open the auth page and exit immediately (non-blocking).
    """
    if is_authenticated():
        return
    _not_logged_in_exit(open_browser=True)


@cli.command()
@click.option("--project_name", required=False)
@click.option("--project_id", required=False)
def pull(project_name, project_id):
    _ensure_auth_or_launch_login()

    payload = {"project_name": project_name, "project_id": project_id}
    try:
        response = requests.post(f"{API_URL}/pull", json=payload, headers=get_auth_headers(), stream=True, timeout=300)
    except requests.RequestException as e:
        click.echo(f"Network error: {e}")
        sys.exit(1)

    stream_sse_response(response)


@cli.command()
@click.argument("params", nargs=-1)
def run(params):
    _ensure_auth_or_launch_login()

    payload = {
        "files": {},
        "params": dict(p.split("=", 1) for p in params if "=" in p),
    }

    # Read files from project directory
    for fname in ["generation.py", "reward_function.py", "dataset.jsonl", "config.json", "project.toml"]:
        file_path = f"./project/{fname}"
        try:
            with open(file_path, "r") as f:
                payload["files"][fname] = f.read()
        except FileNotFoundError:
            click.echo(f"File not found: {file_path}")
            click.echo("Run \033[1mreinforceenow start\033[0m first to initialize project files.")
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading {file_path}: {e}")
            sys.exit(1)

    try:
        response = requests.post(f"{API_URL}/run", json=payload, headers=get_auth_headers(), stream=True, timeout=300)
    except requests.RequestException as e:
        click.echo(f"Network error: {e}")
        sys.exit(1)

    stream_sse_response(response)


@cli.command()
@click.option("--run_id", required=True)
def stop(run_id):
    _ensure_auth_or_launch_login()

    try:
        response = requests.post(f"{API_URL}/stop", json={"run_id": run_id}, headers=get_auth_headers(), timeout=60)
        click.echo(response.text)
    except requests.RequestException as e:
        click.echo(f"Network error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
