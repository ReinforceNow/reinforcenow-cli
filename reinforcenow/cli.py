# cli.py
# ReinforceNow CLI with non-blocking login by default and clear UX

import json
import sys
import os
import shutil
import secrets
from pathlib import Path

import click
import requests

from reinforcenow.auth import (
    is_authenticated,
    get_auth_headers,
    login_flow,
    load_credentials,
    CREDS_FILE,
    get_active_org_from_config,
    set_active_org,
)
from reinforcenow.utils import stream_sse_response

# API URL - can be overridden with environment variable
API_URL = os.environ.get("REINFORCENOW_API_URL", "http://localhost:3000/api")


def get_template_dir():
    """Get the path to the bundled templates directory"""
    return Path(__file__).parent / "templates"


def generate_cuid():
    """
    Generate a CUID-like identifier similar to Prisma's @default(cuid()).
    Format: lowercase alphanumeric string starting with 'c' followed by 24 random characters.
    """
    # CUID format: c + timestamp (8 chars) + counter (4 chars) + fingerprint (4 chars) + random (8 chars)
    # Simplified version: c + 24 random alphanumeric chars
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    random_part = ''.join(secrets.choice(chars) for _ in range(24))
    return f'c{random_part}'


def get_active_organization():
    """
    Get the active organization ID.
    Priority: CLI config > credentials file
    Returns the organization_id or None if not available.
    """
    try:
        # First, check if user has manually set an active org in CLI config
        config_org = get_active_org_from_config()
        if config_org:
            return config_org

        # Fallback to credentials file's organization_id
        creds = load_credentials()
        return creds.get("organization_id")

    except Exception:
        return None


def require_auth():
    """
    Authentication guard decorator/function that ensures user is authenticated.
    If not authenticated, automatically starts the device login flow and waits for completion.

    This provides a seamless authentication experience - users don't need to run
    'reinforcenow login' separately, the CLI handles it automatically.
    """
    if is_authenticated():
        return  # Already authenticated, continue

    # Not authenticated - start device login flow and wait
    click.echo("\033[1m🔐 Authentication required\033[0m\n")
    click.echo("Starting authentication flow...\n")

    # Start device login flow with wait=True to block until complete
    exit_code = login_flow(wait=True, force=False)

    if exit_code != 0:
        click.echo("\n\033[91mAuthentication failed or was cancelled.\033[0m")
        click.echo("Your command cannot proceed without authentication.\n")
        sys.exit(1)

    # Verify authentication succeeded
    if not is_authenticated():
        click.echo("\n\033[91mAuthentication verification failed.\033[0m")
        click.echo("Please try running \033[1mreinforceenow login\033[0m manually.\n")
        sys.exit(1)

    click.echo("\n\033[1m✓ Authentication successful!\033[0m")
    click.echo("Continuing with your command...\n")


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
def status():
    """Check authentication status."""
    if is_authenticated():
        try:
            creds = load_credentials()
            click.echo("\033[1mAuthenticated\033[0m")
            click.echo(f"Credentials file: {CREDS_FILE}")
            click.echo(f"Organization ID: {creds.get('organization_id', 'N/A')}")
            click.echo(f"User ID: {creds.get('user_id', 'N/A')}")
        except Exception as e:
            click.echo(f"\033[1mAuthenticated\033[0m (but could not read details: {e})")
    else:
        click.echo("\033[1mNot authenticated\033[0m")
        click.echo("Run \033[1mreinforceenow login\033[0m to authenticate.")


@cli.command()
def logout():
    """Clear authentication credentials."""
    if CREDS_FILE.exists():
        try:
            CREDS_FILE.unlink()
            click.echo("\033[1mLogged out.\033[0m Credentials removed.")
        except Exception as e:
            click.echo(f"Could not remove credentials: {e}")
    else:
        click.echo("Already logged out - no credentials found.")


@cli.command()
def start():
    """Initialize a quickstart tutorial project with sentiment analysis example."""
    # Check authentication first to fetch organization
    require_auth()

    project_dir = Path("./project")
    dataset_dir = Path("./dataset")
    template_dir = get_template_dir() / "start"  # Use start subfolder for quickstart

    # Generate new IDs for project and dataset
    project_id = generate_cuid()
    dataset_id = generate_cuid()

    # Fetch active organization
    organization_id = get_active_organization()
    if not organization_id:
        click.echo("\033[91mError: Could not fetch active organization.\033[0m")
        click.echo("Please ensure you're logged in and have an active organization.")
        sys.exit(1)

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

    click.echo("\033[1m🚀 Initializing quickstart tutorial project...\033[0m")
    click.echo("This will create a sentiment analysis example to get you started.\n")

    success_count = 0
    failed_files = []

    # Copy project files
    for filename in project_files:
        source = template_dir / filename
        destination = project_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  ✓ Created project/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  ✗ Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  ✗ Error copying {filename}: {e}")
            failed_files.append(filename)

    # Copy dataset files
    for filename in dataset_files:
        source = template_dir / filename
        destination = dataset_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  ✓ Created dataset/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  ✗ Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  ✗ Error copying {filename}: {e}")
            failed_files.append(filename)

    # Summary
    total_files = len(project_files) + len(dataset_files)
    click.echo(f"\n\033[1m✅ Quickstart project initialized!\033[0m")
    click.echo(f"Created: {success_count}/{total_files} files")

    if failed_files:
        click.echo(f"Failed: {', '.join(failed_files)}")
        return

    # Update config.json with generated IDs and organization
    config_file = project_dir / "config.json"
    try:
        with open(config_file, "r") as f:
            config = json.load(f)

        # Update with generated values
        config["project_id"] = project_id
        config["dataset_id"] = dataset_id
        config["organization_id"] = organization_id

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Get project details from updated config
        project_name = config.get("project_name", "GSM8K Project")
        dataset_name = config.get("dataset_name", "GSM8K Dataset")

        click.echo(f"\n\033[1m✓ Configuration updated:\033[0m")
        click.echo(f"  Project Name: {project_name}")
        click.echo(f"  Project ID: {project_id}")
        click.echo(f"  Dataset Name: {dataset_name}")
        click.echo(f"  Dataset ID: {dataset_id}")
        click.echo(f"  Organization ID: {organization_id}")
    except Exception as e:
        click.echo(f"\n\033[91mWarning: Could not update config.json: {e}\033[0m")

    click.echo(f"\n\033[1m📁 Project structure:\033[0m")
    click.echo(f"  ./project/     - Your RLHF project files")
    click.echo(f"  ./dataset/     - Training and validation data")
    click.echo(f"\n\033[1m📚 Quickstart tutorial:\033[0m")
    click.echo(f"  This example trains a sentiment analysis model using RLHF.")
    click.echo(f"  The dataset includes movie/product reviews with labels.")
    click.echo(f"\n\033[1m🎯 Next steps:\033[0m")
    click.echo(f"  1. Review the example files in ./project/ and ./dataset/")
    click.echo(f"  2. Run \033[1mreinforceenow run\033[0m to start training")
    click.echo(f"  3. Check out the docs at https://docs.reinforcenow.ai")


@cli.command()
def orgs():
    """List and switch between organizations."""
    require_auth()

    click.echo("\n\033[1mFetching your organizations...\033[0m\n")

    # Fetch organizations from API
    try:
        response = requests.get(
            f"{API_URL}/auth/organizations",
            headers=get_auth_headers(),
            timeout=10
        )

        if response.status_code != 200:
            click.echo(f"\033[91mError: Could not fetch organizations (status {response.status_code})\033[0m")
            sys.exit(1)

        data = response.json()
        organizations = data.get("organizations", [])

        if not organizations:
            click.echo("\033[91mNo organizations found.\033[0m")
            sys.exit(1)

        # Get current active org
        current_org_id = get_active_organization()

        # Build display strings
        click.echo("\033[1mYour organizations:\033[0m\n")
        for idx, org in enumerate(organizations, 1):
            org_id = org.get("id")
            org_name = org.get("name")
            org_role = org.get("role")
            is_current = org_id == current_org_id

            if is_current:
                click.echo(f"  \033[92m▶ {idx}. {org_name} (role: {org_role}) [CURRENT]\033[0m")
            else:
                click.echo(f"    {idx}. {org_name} (role: {org_role})")

        # Interactive selection with arrow keys or number input
        click.echo()
        click.echo("\033[1mSelect organization:\033[0m Use ↑/↓ arrow keys and Enter, or type a number")

        # Get current selection index (start with current org)
        current_idx = next((i for i, org in enumerate(organizations) if org["id"] == current_org_id), 0)
        selected_idx = current_idx

        # Try to use arrow key selection (Unix-only), fall back to number input
        arrow_keys_available = False
        try:
            import tty
            import termios
            arrow_keys_available = True
        except ImportError:
            pass

        if arrow_keys_available and sys.stdin.isatty():
            try:
                # Try to save terminal settings (will fail if stdin is redirected/piped)
                old_settings = None
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                except (OSError, termios.error, Exception) as e:
                    # stdin is not a real terminal (piped, redirected, etc.)
                    arrow_keys_available = False
                    # Don't print error, just fall through to number input

                if old_settings is None:
                    raise OSError("Terminal not available")

                try:
                    tty.setcbreak(sys.stdin.fileno())

                    while True:
                        # Clear previous line and show current selection
                        sys.stdout.write(f"\r\033[K> {organizations[selected_idx]['name']}")
                        sys.stdout.flush()

                        # Read key
                        char = sys.stdin.read(1)

                        # Handle Enter (select)
                        if char in ('\n', '\r'):
                            break

                        # Handle Escape sequences (arrow keys)
                        elif char == '\x1b':
                            next1 = sys.stdin.read(1)
                            if next1 == '[':
                                next2 = sys.stdin.read(1)
                                if next2 == 'A':  # Up arrow
                                    selected_idx = max(0, selected_idx - 1)
                                elif next2 == 'B':  # Down arrow
                                    selected_idx = min(len(organizations) - 1, selected_idx + 1)

                        # Handle q to quit
                        elif char in ('q', 'Q'):
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            click.echo("Cancelled.")
                            return

                        # Handle number input (1-9)
                        elif char.isdigit():
                            num = int(char)
                            if 1 <= num <= len(organizations):
                                selected_idx = num - 1
                                break

                    # Restore terminal settings
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()

                except Exception as e:
                    # Restore terminal on error
                    try:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    except:
                        pass
                    # Re-raise to trigger fallback
                    raise

            except (OSError, ValueError, termios.error, Exception) as e:
                # Terminal doesn't support raw mode or other error, fall back silently
                arrow_keys_available = False

        if not arrow_keys_available:
            # Fallback to simple number input
            click.echo("(Arrow keys not available - enter a number instead)")

            try:
                selection = click.prompt(
                    "Enter number (or 'q' to cancel)",
                    type=str,
                    default="",
                    show_default=False
                )
            except (OSError, EOFError):
                # stdin not available, cannot get input
                click.echo("\n\033[91mError: Cannot read input in non-interactive mode.\033[0m")
                click.echo("Please run this command in an interactive terminal.")
                sys.exit(1)

            if not selection or selection.lower() in ('q', 'quit', 'cancel'):
                click.echo("Cancelled.")
                return

            try:
                selected_idx = int(selection) - 1
                if selected_idx < 0 or selected_idx >= len(organizations):
                    click.echo("\033[91mInvalid selection.\033[0m")
                    sys.exit(1)
            except ValueError:
                click.echo("\033[91mInvalid input. Please enter a number.\033[0m")
                sys.exit(1)

        # Apply selection
        selected_org = organizations[selected_idx]
        selected_org_id = selected_org["id"]
        selected_org_name = selected_org["name"]

        # Don't switch if already active
        if selected_org_id == current_org_id:
            click.echo(f"\n\033[1mℹ Already using organization:\033[0m {selected_org_name}")
            return

        # Set as active organization
        set_active_org(selected_org_id)

        click.echo(f"\n\033[1m✓ Switched to organization:\033[0m {selected_org_name}")
        click.echo(f"  Organization ID: {selected_org_id}")

    except requests.RequestException as e:
        click.echo(f"\033[91mNetwork error: {e}\033[0m")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\033[91mError: {e}\033[0m")
        sys.exit(1)


@cli.command()
def new():
    """Create a blank project template for custom use cases."""
    # Check authentication first to fetch organization
    require_auth()

    # Prompt for project name
    click.echo("\033[1m📦 Creating new blank project...\033[0m\n")
    project_name = click.prompt("\033[1mProject name\033[0m", type=str, default="My RLHF Project")

    project_dir = Path("./project")
    dataset_dir = Path("./dataset")
    template_dir = get_template_dir() / "new"  # Use new subfolder for blank template

    # Generate new IDs for project and dataset
    project_id = generate_cuid()
    dataset_id = generate_cuid()

    # Fetch active organization
    organization_id = get_active_organization()
    if not organization_id:
        click.echo("\033[91mError: Could not fetch active organization.\033[0m")
        click.echo("Please ensure you're logged in and have an active organization.")
        sys.exit(1)

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

    click.echo()

    success_count = 0
    failed_files = []

    # Copy project files
    for filename in project_files:
        source = template_dir / filename
        destination = project_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  ✓ Created project/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  ✗ Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  ✗ Error copying {filename}: {e}")
            failed_files.append(filename)

    # Copy dataset files
    for filename in dataset_files:
        source = template_dir / filename
        destination = dataset_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  ✓ Created dataset/{filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  ✗ Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  ✗ Error copying {filename}: {e}")
            failed_files.append(filename)

    # Summary
    total_files = len(project_files) + len(dataset_files)
    click.echo(f"\n\033[1m✅ Project initialized!\033[0m")
    click.echo(f"Created: {success_count}/{total_files} files")

    if failed_files:
        click.echo(f"Failed: {', '.join(failed_files)}")
        return

    # Update config.json with generated IDs and organization
    config_file = project_dir / "config.json"
    try:
        with open(config_file, "r") as f:
            config = json.load(f)

        # Update with generated values
        config["project_id"] = project_id
        config["project_name"] = project_name
        config["dataset_id"] = dataset_id
        config["organization_id"] = organization_id

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        click.echo(f"\n\033[1m✓ Configuration updated:\033[0m")
        click.echo(f"  Project Name: {project_name}")
        click.echo(f"  Project ID: {project_id}")
        click.echo(f"  Dataset ID: {dataset_id}")
        click.echo(f"  Organization ID: {organization_id}")
    except Exception as e:
        click.echo(f"\n\033[91mWarning: Could not update config.json: {e}\033[0m")

    click.echo(f"\n\033[1m📁 Project structure:\033[0m")
    click.echo(f"  ./project/     - Your RLHF project files")
    click.echo(f"  ./dataset/     - Training and validation data")
    click.echo(f"\n\033[1m🎯 Next steps:\033[0m")
    click.echo(f"  1. Edit files in ./project/ and ./dataset/")
    click.echo(f"  2. Implement your custom generation and reward logic")
    click.echo(f"  3. Add your training data to ./dataset/")
    click.echo(f"  4. Run \033[1mreinforceenow run\033[0m to start training")




@cli.command()
@click.argument("run_id", required=False, default=None)
@click.option("--pull-type", type=click.Choice(["dataset", "project", "run"]), default="run", help="Type of pull: dataset, project, or run (default)")
@click.option("--dataset-id", help="Dataset ID")
@click.option("--project-id", help="Project ID")
@click.option("--run-id", "run_id_option", help="Run ID (alternative to positional argument)")
@click.option("--dataset-version-id", help="Specific dataset version")
@click.option("--project-version-id", help="Specific project version")
@click.option("--update-config", is_flag=True, default=True, help="Update config.json with pulled IDs (default: true)")
def pull(run_id, pull_type, dataset_id, project_id, run_id_option, dataset_version_id, project_version_id, update_config):
    """
    Pull files from S3.

    Pull by run ID (easiest - use positional argument):
      reinforcenow pull run_xyz123
      reinforcenow pull --run-id run_xyz123

    Pull latest project or dataset (requires both IDs):
      reinforcenow pull --pull-type project --project-id proj_abc --dataset-id data_xyz
      reinforcenow pull --pull-type dataset --project-id proj_abc --dataset-id data_xyz

    Pull specific version (requires parent ID):
      reinforcenow pull --project-id proj_abc --project-version-id ver_123
      reinforcenow pull --dataset-id data_xyz --dataset-version-id ver_456
    """
    require_auth()

    # Use positional run_id if provided, otherwise fall back to --run-id option
    if run_id_option:
        run_id = run_id_option

    # Validate at least one ID is provided
    if not any([run_id, project_id, dataset_id, project_version_id, dataset_version_id]):
        click.echo("Error: Must provide at least one ID (run ID, project ID, dataset ID, or version ID)")
        click.echo("Examples:")
        click.echo("  reinforcenow pull run_xyz123")
        click.echo("  reinforcenow pull --run-id run_xyz123")
        click.echo("  reinforcenow pull --project-id proj_abc --dataset-id data_xyz")
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
    require_auth()

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

    # Validate configuration parameters
    params = config.get("params", {})

    # Validate GPUs (max 8)
    gpus = params.get("gpus", "0")
    try:
        # Parse GPU count - can be a number string like "8" or comma-separated like "0,1,2"
        if isinstance(gpus, str):
            gpu_count = int(gpus) if gpus.isdigit() else len(gpus.split(","))
        elif isinstance(gpus, list):
            gpu_count = len(gpus)
        else:
            gpu_count = int(gpus)

        if gpu_count > 8:
            click.echo(f"\n\033[91mError: Maximum 8 GPUs allowed, but {gpu_count} requested.\033[0m")
            click.echo("For larger GPU configurations, please contact us at support@reinforcenow.com")
            sys.exit(1)
    except Exception:
        pass  # If we can't parse, let backend handle it

    # Validate advantage estimator
    adv_estimator = params.get("algorithm.adv_estimator", "gae")
    valid_estimators = ["gae", "grpo", "reinforce"]
    if adv_estimator not in valid_estimators:
        click.echo(f"\n\033[91mError: Invalid advantage estimator '{adv_estimator}'.\033[0m")
        click.echo(f"Allowed values: {', '.join(valid_estimators)}")
        sys.exit(1)

    # Validate model
    model = params.get("model", "")
    valid_models = ["qwen3-8b", "glm4-9b", "qwen3-30b-a3b", "llama-3.2-3b"]
    if model and model not in valid_models:
        click.echo(f"\n\033[91mError: Invalid model '{model}'.\033[0m")
        click.echo(f"Allowed models: {', '.join(valid_models)}")
        sys.exit(1)

    # Extract required fields
    project_id = config.get("project_id", "your-project-id")
    dataset_id = config.get("dataset_id", "your-dataset-id")

    # Get organization_id from config or fall back to authenticated user's org
    organization_id = config.get("organization_id")
    if not organization_id or organization_id == "your-organization-id":
        organization_id = get_active_organization()
        if not organization_id:
            click.echo("Error: Could not determine organization ID")
            click.echo("Please ensure you're logged in and have an active organization.")
            sys.exit(1)

    # Check if using placeholder IDs
    has_placeholders = (project_id == "your-project-id" or dataset_id == "your-dataset-id")

    if has_placeholders:
        click.echo("\n\033[1mNo project/dataset IDs found. New project will be created...\033[0m")

    click.echo(f"\n\033[1mUploading files...\033[0m")
    click.echo(f"Project ID: {project_id}")
    click.echo(f"Dataset ID: {dataset_id}\n")

    # Prepare multipart form data for upload
    # Note: Backend will create new project and dataset versions automatically
    files = []
    file_handles = []  # Track file handles for cleanup
    data = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "organization_id": organization_id,
    }

    # Add project files (generation.py is optional)
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
        # Get auth headers but remove Content-Type for multipart upload
        headers = get_auth_headers()
        headers.pop('Content-Type', None)  # Let requests set multipart boundary

        upload_response = requests.post(
            f"{API_URL}/training/submit",
            data=data,
            files=files,
            headers=headers,
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
@click.argument("run_id", required=False, default=None)
@click.option("--run-id", "run_id_option", help="Run ID to stop (alternative to positional argument)")
def stop(run_id, run_id_option):
    """
    Stop a training run.

    Examples:
      reinforcenow stop cmgoe8arz000b025oycviytgu
      reinforcenow stop --run-id cmgoe8arz000b025oycviytgu
    """
    require_auth()

    # Use positional run_id if provided, otherwise fall back to --run-id option
    if run_id_option:
        run_id = run_id_option

    if not run_id:
        click.echo("Error: Run ID is required")
        click.echo("Usage: reinforcenow stop <run_id>")
        sys.exit(1)

    click.echo(f"\n\033[1mStopping training run...\033[0m")
    click.echo(f"Run ID: {run_id}\n")

    try:
        response = requests.post(
            f"{API_URL}/training/stop",
            json={"run_id": run_id},
            headers=get_auth_headers(),
            timeout=60
        )

        if response.status_code != 200:
            click.echo(f"\033[91mError: Request failed with status {response.status_code}\033[0m")
            click.echo(response.text)
            sys.exit(1)

        result = response.json()

        if result.get("success"):
            click.echo("\033[1m✓ Training run stopped successfully\033[0m\n")
            click.echo(f"Run ID: {result.get('run_id')}")
            click.echo(f"Status: {result.get('status')}")
            click.echo(f"Duration: {result.get('duration_minutes', 0)} minutes")
            click.echo(f"Charged amount: ${result.get('charged_amount', 0):.2f}")
            if result.get('pod_stopped'):
                click.echo("Pod stopped: Yes")
        else:
            click.echo(f"\033[91mFailed to stop training run\033[0m")
            click.echo(f"Message: {result.get('message', 'Unknown error')}")
            sys.exit(1)

    except requests.RequestException as e:
        click.echo(f"\033[91mNetwork error: {e}\033[0m")
        sys.exit(1)
    except json.JSONDecodeError:
        click.echo(f"\033[91mError: Invalid JSON response from server\033[0m")
        click.echo(response.text)
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
    require_auth()

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
