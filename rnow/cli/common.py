# reinforcenow/cli/common.py

import json

import click

from rnow.cli import auth


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
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def require_auth():
    """Ensure authenticated."""
    if not auth.is_authenticated():
        raise click.ClickException("Not authenticated. Run 'reinforcenow login'")
