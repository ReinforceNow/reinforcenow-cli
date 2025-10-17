# reinforcenow/cli/auth.py

import json
from pathlib import Path
from typing import Dict, Optional

import click
import platformdirs

# Use proper XDG paths for config and data
APP_NAME = "reinforcenow"
APP_AUTHOR = "ReinforceNow"

# Config directory for user preferences
CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))
CONFIG_FILE = CONFIG_DIR / "config.json"

# Data directory for credentials (more secure location)
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
CREDS_FILE = DATA_DIR / "credentials.json"

# Note: For enhanced security, consider using keyring library to store
# credentials in the OS keychain. Current implementation uses file-based
# storage with restricted permissions.


def is_authenticated() -> bool:
    """Check if authenticated."""
    try:
        with open(CREDS_FILE) as f:
            return "api_key" in json.load(f)
    except:
        return False


def get_auth_headers() -> Dict[str, str]:
    """Get auth headers."""
    try:
        with open(CREDS_FILE) as f:
            creds = json.load(f)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {creds['api_key']}"
            }
    except:
        raise click.ClickException("Not authenticated. Run 'reinforcenow login'")


def get_active_org_from_config() -> Optional[str]:
    """Get active organization."""
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f).get("active_organization_id")
    except:
        return None


def set_active_organization(org_id: str) -> None:
    """Set active organization."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    except:
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