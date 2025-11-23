# reinforcenow/cli/commands.py

import json
import time
import uuid
import webbrowser
from pathlib import Path

import click
import requests
import yaml
from pydantic import ValidationError

from rnow import models
from rnow.cli import auth
from rnow.cli.common import require_auth, get_active_organization


# Simple session for API calls
session = requests.Session()
session.headers["User-Agent"] = "ReinforceNow-CLI/1.0"


def api_request(method: str, endpoint: str, base_url: str = None, authenticated: bool = True, **kwargs):
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
    base_url = ctx.obj.get('api_url', 'https://www.reinforcenow.ai/api')

    if not force and auth.is_authenticated():
        click.echo(click.style("✓ Already authenticated", fg="green"))
        click.echo("Use --force to re-authenticate")
        return

    # Get device code
    try:
        response = api_request("post", "/auth/device/code", base_url,
                              json={"client_id": "cli"}, authenticated=False)
        response.raise_for_status()
        device = models.DeviceCode(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid response from server: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to initiate login: {e}")

    # Construct the full URL with user_code parameter
    verification_url = f"{device.verification_uri}?user_code={device.user_code}"

    click.echo(f"\n{click.style('Opening browser:', fg='cyan')} {verification_url}")
    click.echo(f"{click.style('Enter code:', fg='cyan')} {click.style(device.user_code, bold=True)}\n")
    webbrowser.open(verification_url)

    # Poll for token
    start = time.time()
    with click.progressbar(length=device.expires_in//device.interval, label='Waiting for authentication', show_pos=False) as bar:
        while time.time() - start < device.expires_in:
            time.sleep(device.interval)
            bar.update(1)

            try:
                resp = api_request("post", "/auth/device/token", base_url,
                                 json={"device_code": device.device_code}, authenticated=False)
                data = resp.json()
            except requests.RequestException as e:
                raise click.ClickException(f"Network error: {e}")

            if resp.status_code == 200:
                try:
                    token = models.Token(**data)
                except ValidationError as e:
                    raise click.ClickException(f"Invalid token response: {e}")

                # Save credentials
                auth.DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(auth.CREDS_FILE, "w") as f:
                    json.dump({"api_key": token.access_token, "organization_id": token.organization_id}, f)
                auth.CREDS_FILE.chmod(0o600)

                bar.finish()
                click.echo(click.style("\n✓ Login successful!", fg="green", bold=True))
                return

            try:
                error = models.TokenError(**data)
            except ValidationError:
                raise click.ClickException(f"Unexpected response: {data}")

            if error.error != "authorization_pending":
                bar.finish()
                raise click.ClickException(f"Authentication failed: {error.error}")

    raise click.ClickException("Authentication timed out")


@click.command()
def logout():
    """Logout from ReinforceNow."""
    auth.logout()


@click.command()
def status():
    """Check authentication status."""
    if auth.is_authenticated():
        click.echo(click.style("✓ Authenticated", fg="green"))
        org_id = get_active_organization()
        if org_id:
            click.echo(f"Organization: {org_id}")
    else:
        click.echo(click.style("✗ Not authenticated", fg="red"))
        raise click.ClickException("Run 'reinforcenow login' to authenticate")


# ========== Org Commands ==========

@click.group()
def orgs():
    """Manage organizations."""
    pass


@orgs.command("list")
@click.pass_context
def orgs_list(ctx):
    """List all available organizations."""
    base_url = ctx.obj.get('api_url', 'https://www.reinforcenow.ai/api')

    try:
        response = api_request("get", "/auth/organizations", base_url)
        response.raise_for_status()
        orgs = models.Organizations(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid organization data: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to fetch organizations: {e}")

    if not orgs.organizations:
        click.echo(click.style("No organizations found", fg="yellow"))
        return

    click.echo(click.style("Organizations:", bold=True))
    for org in orgs.organizations:
        if org.id == orgs.active_organization_id:
            mark = click.style("✓", fg="green")
            name = click.style(org.name, bold=True)
        else:
            mark = " "
            name = org.name
        click.echo(f"  [{mark}] {name} ({org.id}) - {org.role.value}")


@orgs.command("select")
@click.argument("org_id", required=True)
def orgs_select(org_id: str):
    """Set active organization by ID."""
    require_auth()
    auth.set_active_organization(org_id)
    click.echo(click.style(f"✓ Active organization set to: {org_id}", fg="green"))


# ========== Project Commands ==========

@click.command()
@click.option("--template", "-t", type=click.Choice(["start", "new", "blank", "sft", "rl-single", "rl-multi", "rl-tools"]), default="start", help="Project template to use")
@click.option("--name", "-n", help="Project name (will prompt if not provided)")
def start(template: str, name: str):
    """Initialize a new ReinforceNow project."""
    require_auth()

    import shutil
    from pathlib import Path

    project_name = name or click.prompt("Project name", default="My RLHF Project", type=str)

    # Create project directory in current location
    project_dir = Path(".")

    # Map "start" to "rl-single"
    actual_template = "rl-single" if template == "start" else template

    # Copy template files if template is specified (all except blank)
    if actual_template != "blank":
        template_dir = Path(__file__).parent.parent / "templates" / actual_template
        if template_dir.exists():
            # Copy all template files to current directory
            for file in template_dir.iterdir():
                if file.is_file():
                    shutil.copy2(file, project_dir / file.name)
                    click.echo(f"  Created {file.name}")
        else:
            click.echo(click.style(f"Warning: Template '{template}' not found, using blank template", fg="yellow"))

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
        config_data['project_id'] = project_id
        config_data['project_name'] = project_name
        config_data['dataset_id'] = dataset_id
        config_data['organization_id'] = org_id

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    else:
        # Create new config for blank template
        config = models.ProjectConfig(
            project_id=project_id,
            project_name=project_name,
            dataset_id=dataset_id,
            dataset_type=models.DatasetType.RL,
            organization_id=org_id,
            params=models.TrainingParams(
                model="meta-llama/Llama-3.2-1B-Instruct",
                qlora_rank=32,
                batch_size=8,
                num_epochs=3,
                learning_rate=1e-4,
                max_steps=None,
                eval_step=100,
                save_step=1,
                loss_fn="ppo",
                adv_estimator="grpo"
            )
        )

        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)
        click.echo(f"  Created config.yml")

    click.echo(click.style(f"\n✓ Created project: {project_name}", fg="green"))
    click.echo(f"\nProject ID: {project_id}")
    click.echo(f"Dataset ID: {dataset_id}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Add training data to train.jsonl")
    click.echo(f"  3. Run 'rnow run' to start training")


@click.command()
@click.option("--dir", "-d", default=".",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help="Directory containing project files (default: current directory)")
@click.option("--name", "-n", default=None,
              help="Custom name for the training run (default: auto-generated)")
@click.pass_context
def run(ctx, dir: Path, name: str):
    """Submit project for training on ReinforceNow platform."""
    require_auth()
    base_url = ctx.obj.get('api_url', 'https://www.reinforcenow.ai/api')

    # Load and validate config
    config_yml = dir / "config.yml"
    config_json = dir / "config.json"

    if config_yml.exists():
        try:
            with open(config_yml) as f:
                config = models.ProjectConfig(**yaml.safe_load(f))
        except FileNotFoundError:
            raise click.ClickException(f"Config file not found in {dir}")
        except ValidationError as e:
            raise click.ClickException(f"Invalid project config: {e}")
        except yaml.YAMLError as e:
            raise click.ClickException(f"Invalid YAML in config file: {e}")
    elif config_json.exists():
        try:
            with open(config_json) as f:
                config = models.ProjectConfig(**json.load(f))
        except ValidationError as e:
            raise click.ClickException(f"Invalid project config: {e}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in config file: {e}")
    else:
        raise click.ClickException(f"No config.yml or config.json found in {dir}")

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
            empty_files.append(f"  • {file_name} is empty - please add training data")

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

    # Upload files
    files = []

    # Add config file
    if config_yml.exists():
        files.append(("config_yml", ("config.yml", open(config_yml, "rb"), "application/octet-stream")))
    elif config_json.exists():
        files.append(("config_json", ("config.json", open(config_json, "rb"), "application/octet-stream")))

    # Add required files
    for file_name, path in required_files.items():
        files.append((file_name.replace(".", "_"), (file_name, open(path, "rb"), "application/octet-stream")))

    # Add optional files (all in the same directory now)
    optional_files = {
        "env.py": dir / "env.py",
        "project.toml": dir / "project.toml"
    }

    for file_name, path in optional_files.items():
        if path.exists():
            files.append((file_name.replace(".", "_"), (file_name, open(path, "rb"), "application/octet-stream")))

    # Show submission summary
    click.echo(click.style("\nSubmitting training job:", bold=True))
    click.echo(f"  Project: {config.project_name}")
    if name:
        click.echo(f"  Run Name: {name}")
    click.echo(f"  Model: {config.params.model if config.params else 'default'}")
    click.echo(f"  Files: {len(files)} files ready for upload")

    # For multipart, we need to omit Content-Type so requests sets the boundary
    headers = auth.get_auth_headers()
    headers.pop("Content-Type", None)

    click.echo("\n" + click.style("Uploading files...", fg="yellow"))

    # Include custom run name if provided
    submit_data = {
        "project_id": config.project_id,
        "dataset_id": config.dataset_id,
        "organization_id": config.organization_id
    }
    if name:
        submit_data["run_name"] = name

    try:
        response = session.post(
            f"{base_url}/training/submit",
            data=submit_data,
            files=files,
            headers=headers
        )
    finally:
        # Close files
        for _, (_, fh, _) in files:
            fh.close()

    if response.status_code != 200:
        raise click.ClickException(f"Training submission failed: {response.text}")

    click.echo(click.style("✓ Files uploaded successfully", fg="green"))
    click.echo("\n" + click.style("Training output:", bold=True))

    # Stream output
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            click.echo("  " + line[6:])


@click.command()
@click.argument("run_id", required=True)
@click.confirmation_option(prompt="Are you sure you want to stop this training run?")
@click.pass_context
def stop(ctx, run_id: str):
    """Stop an active training run.

    Requires the RUN_ID obtained from 'rnow run' command.
    """
    base_url = ctx.obj.get('api_url', 'https://www.reinforcenow.ai/api')

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