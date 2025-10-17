# reinforcenow/cli/common.py

import json
from pathlib import Path

import click

from reinforcenow.cli import auth

API_URL = "https://www.reinforcenow.ai/api"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def get_active_organization():
    """Get active organization ID (config > credentials)."""
    # First check config
    org_id = auth.get_active_org_from_config()
    if org_id:
        return org_id

    # Fall back to credentials
    try:
        with open(auth.CREDS_FILE) as f:
            return json.load(f).get("organization_id")
    except:
        return None


def require_auth():
    """Ensure authenticated."""
    if not auth.is_authenticated():
        raise click.ClickException("Not authenticated. Run 'reinforcenow login'")