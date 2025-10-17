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

from reinforcenow import models
from reinforcenow.cli import auth
from reinforcenow.cli.common import require_auth, get_active_organization


class API:
    """Simple API wrapper with session management."""

    def __init__(self, base_url: str = "https://www.reinforcenow.ai/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ReinforceNow-CLI/1.0"

    def request(self, method: str, endpoint: str, authenticated: bool = True, **kwargs):
        """Make API request."""
        if authenticated:
            require_auth()
            headers = kwargs.pop("headers", {})
            headers.update(auth.get_auth_headers())
            kwargs["headers"] = headers

        url = f"{self.base_url}{endpoint}"
        return getattr(self.session, method)(url, **kwargs)

    def get(self, endpoint: str, **kwargs):
        """GET request."""
        return self.request("get", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs):
        """POST request."""
        return self.request("post", endpoint, **kwargs)


# Single API instance for the CLI
api = API()


# ========== Auth Commands ==========

@click.command()
@click.option("--force", "-f", is_flag=True, help="Force new login even if already authenticated")
def login(force: bool) -> models.LoginOutput:
    """Login to ReinforceNow platform.

    Uses OAuth device flow for authentication.
    """
    if not force and auth.is_authenticated():
        click.echo(click.style("✓ Already authenticated", fg="green"))
        click.echo("Use --force to re-authenticate")
        # Return existing credentials from file
        try:
            with open(auth.CREDS_FILE) as f:
                creds = json.load(f)
                return models.LoginOutput(
                    access_token=creds.get("api_key", ""),
                    organization_id=creds.get("organization_id")
                )
        except:
            raise click.ClickException("Failed to read existing credentials")

    # Get device code
    try:
        response = api.post("/auth/device/code", json={"client_id": "cli"}, authenticated=False)
        response.raise_for_status()
        device = models.DeviceCode(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid response from server: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to initiate login: {e}")

    click.echo(f"\n{click.style('Opening browser:', fg='cyan')} {device.verification_uri}")
    click.echo(f"{click.style('Enter code:', fg='cyan')} {click.style(device.user_code, bold=True)}\n")
    webbrowser.open(device.verification_uri)

    # Poll for token
    start = time.time()
    with click.progressbar(length=device.expires_in//device.interval, label='Waiting for authentication', show_pos=False) as bar:
        while time.time() - start < device.expires_in:
            time.sleep(device.interval)
            bar.update(1)

            try:
                resp = api.post("/auth/device/token", json={"device_code": device.device_code}, authenticated=False)
                data = resp.json()
            except requests.RequestException as e:
                raise click.ClickException(f"Network error: {e}")

            if resp.status_code == 200:
                try:
                    token = models.Token(**data)
                except ValidationError as e:
                    raise click.ClickException(f"Invalid token response: {e}")

                # Save credentials with secure permissions
                auth.DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(auth.CREDS_FILE, "w") as f:
                    json.dump({"api_key": token.access_token, "organization_id": token.organization_id}, f)
                # Set restrictive permissions (user read/write only)
                auth.CREDS_FILE.chmod(0o600)

                bar.finish()
                click.echo(click.style("\n✓ Login successful!", fg="green", bold=True))
                return models.LoginOutput(access_token=token.access_token, organization_id=token.organization_id)

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
def orgs_list() -> models.Organizations:
    """List all available organizations."""
    try:
        response = api.get("/auth/organizations")
        response.raise_for_status()
        orgs = models.Organizations(**response.json())
    except ValidationError as e:
        raise click.ClickException(f"Invalid organization data: {e}")
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to fetch organizations: {e}")

    if not orgs.organizations:
        click.echo(click.style("No organizations found", fg="yellow"))
        return orgs

    click.echo(click.style("Organizations:", bold=True))
    for org in orgs.organizations:
        if org.id == orgs.active_organization_id:
            mark = click.style("✓", fg="green")
            name = click.style(org.name, bold=True)
        else:
            mark = " "
            name = org.name
        click.echo(f"  [{mark}] {name} ({org.id}) - {org.role.value}")

    return orgs


@orgs.command("select")
@click.argument("org_id", required=True)
def orgs_select(org_id: str):
    """Set active organization by ID."""
    require_auth()
    auth.set_active_organization(org_id)
    click.echo(click.style(f"✓ Active organization set to: {org_id}", fg="green"))


# ========== Project Commands ==========

@click.command()
@click.option("--template", "-t", default="blank", help="Project template to use")
@click.option("--name", "-n", help="Project name (will prompt if not provided)")
def start(template: str, name: str) -> models.ProjectCreateOutput:
    """Initialize a new ReinforceNow project."""
    require_auth()

    project_name = name or click.prompt("Project name", default="My RLHF Project", type=str)
    project_dir = Path("./project")
    dataset_dir = Path("./dataset")

    project_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    project_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())

    config = models.ProjectConfig(
        project_id=project_id,
        project_name=project_name,
        dataset_id=dataset_id,
        dataset_type=models.DatasetType.RL,  # Default to RL
        organization_id=get_active_organization(),
        params=models.TrainingParams(
            model=models.ModelType.QWEN3_8B,
            qlora_rank=32,
            batch_size=32,
            num_epochs=3,
            max_steps=None,
            val_steps=100,  # Default to step-based validation
            save_steps=500,  # Save every 500 steps
            loss_fn=models.LossFunction.PPO,
            adv_estimator=models.AdvantageEstimator.GRPO
            # compute_post_kl defaults to False
            # kl_penalty_coef defaults to 0.01
        )
    )

    config_path = project_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)

    click.echo(click.style(f"✓ Created project: {project_name}", fg="green"))
    click.echo(f"\nProject ID: {project_id}")
    click.echo(f"Dataset ID: {dataset_id}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Add training data to {dataset_dir}/train.jsonl")
    click.echo(f"  2. Implement reward function in {project_dir}/reward_function.py")
    click.echo(f"  3. Run 'reinforcenow run' to start training")

    return models.ProjectCreateOutput(
        project_id=project_id,
        project_name=project_name,
        dataset_id=dataset_id,
        organization_id=config.organization_id,
        config_path=config_path,
        project_dir=project_dir,
        dataset_dir=dataset_dir
    )


@click.command()
@click.option("--project-dir", "-p", default="./project",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help="Project directory containing config and code files")
@click.option("--dataset-dir", "-d", default="./dataset",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help="Dataset directory containing training data")
def run(project_dir: Path, dataset_dir: Path) -> models.TrainingSubmitOutput:
    """Submit project for training on ReinforceNow platform."""
    require_auth()

    # Load and validate config (supports both .yml and .json for backwards compatibility)
    config_yml = project_dir / "config.yml"
    config_json = project_dir / "config.json"

    if config_yml.exists():
        try:
            with open(config_yml) as f:
                config = models.ProjectConfig(**yaml.safe_load(f))
        except FileNotFoundError:
            raise click.ClickException(f"Config file not found in {project_dir}")
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
        raise click.ClickException(f"No config.yml or config.json found in {project_dir}")

    if not config.organization_id:
        config.organization_id = get_active_organization()

    # Validate required files
    required_files = {
        "train.jsonl": dataset_dir / "train.jsonl",
        "reward_function.py": project_dir / "reward_function.py"
    }

    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"  • {name} at {path}")

    if missing_files:
        click.echo(click.style("✗ Required files missing:", fg="red", bold=True))
        for file_msg in missing_files:
            click.echo(file_msg)
        raise click.ClickException("Missing required files for training submission")

    # Upload files (config, required files, and optional files)
    files = []

    # Add config file
    if config_yml.exists():
        files.append(("config_yml", ("config.yml", open(config_yml, "rb"), "application/octet-stream")))
    elif config_json.exists():
        files.append(("config_json", ("config.json", open(config_json, "rb"), "application/octet-stream")))

    # Add required files
    for name, path in required_files.items():
        files.append((name.replace(".", "_"), (name, open(path, "rb"), "application/octet-stream")))

    # Add optional files
    optional_files = {
        "generation.py": project_dir / "generation.py",
        "val.jsonl": dataset_dir / "val.jsonl",
        "project.toml": project_dir / "project.toml"
    }

    for name, path in optional_files.items():
        if path.exists():
            files.append((name.replace(".", "_"), (name, open(path, "rb"), "application/octet-stream")))

    # Show submission summary
    click.echo(click.style("\nSubmitting training job:", bold=True))
    click.echo(f"  Project: {config.project_name}")
    click.echo(f"  Model: {config.params.model.value if config.params else 'default'}")
    click.echo(f"  Files: {len(files)} files ready for upload")

    # For multipart, we need to omit Content-Type so requests sets the boundary
    headers = auth.get_auth_headers()
    headers.pop("Content-Type", None)

    click.echo("\n" + click.style("Uploading files...", fg="yellow"))

    try:
        response = api.session.post(
            f"{api.base_url}/training/submit",
            data={"project_id": config.project_id, "dataset_id": config.dataset_id, "organization_id": config.organization_id},
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

    return models.TrainingSubmitOutput(
        run_id="submitted",
        project_id=config.project_id,
        dataset_id=config.dataset_id,
        status=models.RunStatus.PENDING
    )


@click.command()
@click.argument("run_id", required=True)
@click.confirmation_option(prompt="Are you sure you want to stop this training run?")
def stop(run_id: str) -> models.TrainingStopOutput:
    """Stop an active training run.

    Requires the RUN_ID obtained from 'reinforcenow run' command.
    """
    try:
        click.echo(f"Stopping training run: {run_id}...")
        response = api.post("/training/stop", json={"run_id": run_id})
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to stop training: {e}")

    click.echo(click.style(f"✓ Training run stopped: {run_id}", fg="green"))

    result = models.TrainingStopOutput(
        run_id=run_id,
        status=data.get("status", "stopped"),
        duration_minutes=data.get("duration_minutes"),
        charged_amount=data.get("charged_amount")
    )

    if result.duration_minutes:
        click.echo(f"  Duration: {result.duration_minutes:.1f} minutes")
    if result.charged_amount:
        click.echo(f"  Charged: ${result.charged_amount:.2f}")

    return result