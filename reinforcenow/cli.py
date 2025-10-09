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
    dataset_dir = Path("./dataset")
    template_dir = get_template_dir()

    # Create directories
    project_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    # Project files to copy to ./project/
    project_files = [
        "generation.py",
        "reward_function.py",
        "config.json",
        "project.toml",
    ]

    # Dataset files to copy to ./dataset/
    dataset_files = [
        "train.jsonl",
        "val.jsonl",
    ]

    click.echo("Initializing project with template files...")

    success_count = 0
    failed_files = []

    # Copy project files
    for filename in project_files:
        source = template_dir / filename
        destination = project_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  Created project/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  Error copying {filename}: {e}")
            failed_files.append(filename)

    # Copy dataset files
    for filename in dataset_files:
        source = template_dir / filename
        destination = dataset_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  Created dataset/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  Error copying {filename}: {e}")
            failed_files.append(filename)

    # Summary
    total_files = len(project_files) + len(dataset_files)
    click.echo(f"\n\033[1mProject initialized.\033[0m")
    click.echo(f"Created: {success_count}/{total_files} files")

    if failed_files:
        click.echo(f"Failed: {', '.join(failed_files)}")
        return

    click.echo(f"Project files: ./project/")
    click.echo(f"Dataset files: ./dataset/")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Edit config.json and set your organization_id")
    click.echo(f"  2. Edit files in ./project/ and ./dataset/")
    click.echo(f"  3. Run \033[1mreinforceenow run\033[0m")
    click.echo(f"     (New project/dataset will be created automatically on first run)")


def _ensure_auth_or_launch_login() -> None:
    """
    For commands that require auth: if not authenticated,
    open the auth page and exit immediately (non-blocking).
    """
    if is_authenticated():
        return
    _not_logged_in_exit(open_browser=True)


@cli.command()
@click.option("--pull-type", type=click.Choice(["dataset", "project", "run"]), default="run", help="Type of pull: dataset, project, or run (default)")
@click.option("--dataset-id", help="Dataset ID")
@click.option("--project-id", help="Project ID")
@click.option("--run-id", help="Run ID")
@click.option("--dataset-version-id", help="Specific dataset version")
@click.option("--project-version-id", help="Specific project version")
@click.option("--update-config", is_flag=True, default=True, help="Update config.json with pulled IDs (default: true)")
def pull(pull_type, dataset_id, project_id, run_id, dataset_version_id, project_version_id, update_config):
    """
    Pull files from S3.

    Required: One of --run-id, --project-id, --project-version-id, --dataset-id, or --dataset-version-id

    Examples:
      reinforcenow pull --run-id abc123
      reinforcenow pull --pull-type dataset --dataset-id abc123
      reinforcenow pull --pull-type project --project-id xyz789
      reinforcenow pull --project-version-id ver123
    """
    _ensure_auth_or_launch_login()

    # Validate at least one ID is provided
    if not any([run_id, project_id, dataset_id, project_version_id, dataset_version_id]):
        click.echo("Error: Must provide at least one of: --run-id, --project-id, --project-version-id, --dataset-id, or --dataset-version-id")
        sys.exit(1)

    params = {"pull_type": pull_type}
    if dataset_id:
        params["dataset_id"] = dataset_id
    if project_id:
        params["project_id"] = project_id
    if run_id:
        params["run_id"] = run_id
    if dataset_version_id:
        params["dataset_version_id"] = dataset_version_id
    if project_version_id:
        params["project_version_id"] = project_version_id

    try:
        response = requests.post(f"{API_URL}/projects/pull", json=params, headers=get_auth_headers(), stream=True, timeout=300)
    except requests.RequestException as e:
        click.echo(f"Network error: {e}")
        sys.exit(1)

    stream_sse_response(response)

    # After pull completes, update config.json if requested
    if update_config:
        config_file = Path("./project/config.json")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Update IDs if we have them
                updated = False
                if project_id:
                    config["project_id"] = project_id
                    updated = True
                if dataset_id:
                    config["dataset_id"] = dataset_id
                    updated = True

                # If pulling by run_id, try to get project/dataset IDs from response
                # (This would need the API to return metadata, for now just note the run_id)
                if run_id and not project_id and not dataset_id:
                    # Future: Could make a separate API call to get run metadata
                    pass

                if updated:
                    with open(config_file, "w") as f:
                        json.dump(config, f, indent=2)
                    click.echo(f"\n\033[1m✓ Updated config.json with pulled IDs\033[0m")

            except Exception as e:
                click.echo(f"\nWarning: Could not update config.json: {e}")


@cli.command()
@click.option("--project-dir", default="./project", help="Project directory (default: ./project)")
@click.option("--dataset-dir", default="./dataset", help="Dataset directory (default: ./dataset)")
def run(project_dir, dataset_dir):
    """
    Upload files and start a training run.

    Reads config.json for project/dataset IDs and training parameters.
    Creates new project and dataset versions automatically.

    Example:
      reinforcenow run
      reinforcenow run --project-dir ./my-project --dataset-dir ./my-data
    """
    _ensure_auth_or_launch_login()

    project_path = Path(project_dir)
    dataset_path = Path(dataset_dir)

    # Read config.json
    config_file = project_path / "config.json"
    if not config_file.exists():
        click.echo(f"Error: config.json not found at {config_file}")
        click.echo("Run \033[1mreinforceenow start\033[0m to initialize project files.")
        sys.exit(1)

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except Exception as e:
        click.echo(f"Error reading config.json: {e}")
        sys.exit(1)

    # Extract required fields
    project_id = config.get("project_id", "your-project-id")
    dataset_id = config.get("dataset_id", "your-dataset-id")
    organization_id = config.get("organization_id")
    version = config.get("version", "1.0.0")

    if not organization_id or organization_id == "your-organization-id":
        click.echo("Error: Please set 'organization_id' in config.json")
        click.echo("The organization_id is required to create or update projects.")
        sys.exit(1)

    # Check if using placeholder IDs
    has_placeholders = (project_id == "your-project-id" or dataset_id == "your-dataset-id")

    if has_placeholders:
        click.echo("\n\033[1mNo project/dataset IDs found. New project will be created...\033[0m")

    click.echo(f"\n\033[1mUploading files...\033[0m")
    click.echo(f"Project ID: {project_id}")
    click.echo(f"Dataset ID: {dataset_id}")
    click.echo(f"Version: {version}\n")

    # Prepare multipart form data for upload
    files = []
    file_handles = []  # Track file handles for cleanup
    data = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "version": version,
        "organization_id": organization_id,
    }

    # Add project files
    project_files = ["config.json", "generation.py", "reward_function.py", "project.toml"]
    for fname in project_files:
        file_path = project_path / fname
        if file_path.exists():
            fh = open(file_path, "rb")
            file_handles.append(fh)
            files.append((fname.replace(".", "_"), (fname, fh, "application/octet-stream")))
        else:
            click.echo(f"Warning: Project file not found: {file_path}")

    # Add dataset files
    dataset_files = ["train.jsonl", "val.jsonl"]
    for fname in dataset_files:
        file_path = dataset_path / fname
        if file_path.exists():
            fh = open(file_path, "rb")
            file_handles.append(fh)
            files.append((fname.replace(".", "_"), (fname, fh, "application/octet-stream")))
        elif fname == "train.jsonl":
            click.echo(f"Error: Required file not found: {file_path}")
            sys.exit(1)

    # Upload files and start training (combined in /training/submit)
    try:
        upload_response = requests.post(
            f"{API_URL}/training/submit",
            data=data,
            files=files,
            headers=get_auth_headers(),
            stream=True,
            timeout=300
        )

        # Close file handles
        for fh in file_handles:
            fh.close()

        if upload_response.status_code != 200:
            click.echo(f"Upload failed: {upload_response.status_code}")
            click.echo(upload_response.text)
            sys.exit(1)

        # Parse SSE stream and display progress
        click.echo("\033[1mUploading and starting training...\033[0m\n")

        for line in upload_response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                message = line[6:]  # Remove "data: " prefix
                click.echo(message)

        click.echo("\n\033[1m✓ Training submitted\033[0m\n")

    except requests.RequestException as e:
        # Clean up file handles
        for fh in file_handles:
            try:
                fh.close()
            except:
                pass
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--run-id", required=True, help="Run ID to stop")
def stop(run_id):
    """Stop a training run."""
    _ensure_auth_or_launch_login()

    try:
        response = requests.post(f"{API_URL}/training/stop", json={"run_id": run_id}, headers=get_auth_headers(), timeout=60)
        click.echo(response.text)
    except requests.RequestException as e:
        click.echo(f"Network error: {e}")
        sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option("--output", "-o", default="./models", help="Output directory (default: ./models)")
def download(model_id, output):
    """
    Download a trained model.

    Example:
      reinforcenow download model_abc123
      reinforcenow download model_abc123 --output ./my-models
    """
    _ensure_auth_or_launch_login()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n\033[1mDownloading model {model_id}...\033[0m")

    try:
        # Get download URL from backend
        response = requests.get(
            f"{API_URL}/models/{model_id}/download",
            headers=get_auth_headers(),
            timeout=60
        )

        if response.status_code != 200:
            click.echo(f"Failed to get download URL: {response.status_code}")
            click.echo(response.text)
            sys.exit(1)

        result = response.json()
        download_url = result.get("downloadUrl")
        model_size = result.get("modelSize", 0)
        expires_in = result.get("expiresIn", 3600)

        if not download_url:
            click.echo("Error: No download URL returned")
            sys.exit(1)

        click.echo(f"Model size: {model_size / (1024**3):.2f} GB")
        click.echo(f"Download URL expires in: {expires_in // 60} minutes")
        click.echo(f"Downloading to: {output_dir}/\n")

        # Download file from S3 pre-signed URL
        with requests.get(download_url, stream=True, timeout=None) as r:
            r.raise_for_status()

            # Get filename from Content-Disposition header or use model_id
            filename = f"{model_id}.tar.gz"
            if "Content-Disposition" in r.headers:
                import re
                match = re.search(r'filename="?([^"]+)"?', r.headers["Content-Disposition"])
                if match:
                    filename = match.group(1)

            output_path = output_dir / filename

            # Download with progress bar
            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            click.echo(f"\rProgress: {progress:.1f}%", nl=False)

        click.echo(f"\n\n\033[1m✓ Download complete\033[0m")
        click.echo(f"Model saved to: {output_path}")

    except requests.RequestException as e:
        click.echo(f"\nDownload error: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
