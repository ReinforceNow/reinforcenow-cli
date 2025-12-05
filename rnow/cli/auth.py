# reinforcenow/cli/auth.py

import json
from pathlib import Path

import click

# Simple home directory paths
DATA_DIR = Path.home() / ".reinforcenow"
CREDS_FILE = DATA_DIR / "credentials.json"
CONFIG_FILE = DATA_DIR / "config.json"


def is_authenticated() -> bool:
    """Check if authenticated."""
    try:
        with open(CREDS_FILE) as f:
            return "api_key" in json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def get_auth_headers() -> dict[str, str]:
    """Get auth headers."""
    try:
        with open(CREDS_FILE) as f:
            creds = json.load(f)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {creds['api_key']}",
            }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        raise click.ClickException("Not authenticated. Run 'reinforcenow login'")


def get_active_org_from_config() -> str | None:
    """Get active organization."""
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f).get("active_organization_id")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def set_active_organization(org_id: str) -> None:
    """Set active organization."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    config["active_organization_id"] = org_id

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def logout() -> None:
    """Remove credentials."""
    if CREDS_FILE.exists():
        CREDS_FILE.unlink()
        click.echo("✓ Logged out")
    else:
        click.echo("Not logged in")
