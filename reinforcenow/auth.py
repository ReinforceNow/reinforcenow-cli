# auth.py
# Simplified device login for ReinforceNow CLI

import json
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# === Configuration ===
import os

# Allow override via environment variable for development
AUTH_BASE_URL = os.environ.get("REINFORCENOW_AUTH_URL", "http://localhost:3000")
CLIENT_ID = "cli"
USER_AGENT = "Reinmax-CLI/1.1"

DEVICE_AUTH_URL = f"{AUTH_BASE_URL}/api/auth/device/code"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/auth/device/token"

# Files under ~/.reinmax
CREDS_DIR = Path.home() / ".reinmax"
CREDS_FILE = CREDS_DIR / "credentials"
CONFIG_FILE = CREDS_DIR / "config.json"  # CLI configuration (active org, etc.)


# === Utilities ===
def _ensure_dirs() -> None:
    CREDS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _now_ts() -> int:
    return int(time.time())


def get_cli_config() -> Dict[str, Any]:
    """Get CLI configuration (active organization, etc.)"""
    return _load_json(CONFIG_FILE) or {}


def save_cli_config(config: Dict[str, Any]) -> None:
    """Save CLI configuration"""
    _save_json(CONFIG_FILE, config)


def get_active_org_from_config() -> Optional[str]:
    """Get the active organization ID from CLI config"""
    config = get_cli_config()
    return config.get("active_organization_id")


def set_active_org(org_id: str) -> None:
    """Set the active organization ID in CLI config"""
    config = get_cli_config()
    config["active_organization_id"] = org_id
    save_cli_config(config)


# === Public helpers ===
def save_credentials(creds: Dict[str, Any]) -> None:
    """Save credentials to file."""
    _save_json(CREDS_FILE, creds)


def load_credentials() -> Dict[str, Any]:
    """Load credentials from file."""
    if not CREDS_FILE.exists():
        raise RuntimeError("Not authenticated. Run `reinforcenow login`.")

    data = _load_json(CREDS_FILE)
    if not data or "api_key" not in data:
        raise RuntimeError("Not authenticated. Run `reinforcenow login`.")

    return data


def is_authenticated() -> bool:
    """Check if credentials exist."""
    try:
        load_credentials()
        return True
    except:
        return False


def get_auth_headers() -> Dict[str, str]:
    """Get authorization headers with API key."""
    creds = load_credentials()
    return {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT
    }


# === Device Flow ===
def begin_device_login() -> Optional[Dict[str, Any]]:
    """
    Start device authorization and open the browser.
    Returns device flow data or None on failure.
    """
    try:
        resp = requests.post(
            DEVICE_AUTH_URL,
            json={"client_id": CLIENT_ID},
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Unable to reach auth server: {e}")
        return None

    if resp.status_code != 200:
        print(f"Device authorization failed: {resp.text}")
        return None

    device_data = resp.json()

    # Show user instructions
    print(f"\nOpening authorization page in your browser...")
    print(f"URL: {device_data.get('verification_uri_complete', device_data['verification_uri'])}")
    print(f"Authorization code: \033[1m{device_data['user_code']}\033[0m\n")

    # Try to open the browser
    try:
        verification_url = device_data.get('verification_uri_complete') or device_data['verification_uri']
        webbrowser.open(verification_url)
    except Exception:
        pass

    # Give browser a moment to open
    time.sleep(1)

    # Clear instructions for the user
    print("\033[1mComplete the authorization in your browser to continue.\033[0m")
    print("\033[91mLogin first if you have to.\033[0m")
    print()

    return device_data


def poll_for_api_key(device_code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Poll device token endpoint to get API key directly.
    Returns (credentials_dict, error_code).

    Success: ({api_key, organization_id, user_id}, None)
    Error: (None, error_code)
    """
    try:
        resp = requests.post(
            DEVICE_TOKEN_URL,
            json={"device_code": device_code},
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            timeout=30,
        )

        # Success - got API key!
        if resp.status_code == 200:
            data = resp.json()
            return {
                "api_key": data['access_token'],
                "organization_id": data.get('organization_id'),
                "user_id": data.get('user_id')
            }, None

        # Error response
        elif resp.status_code == 400:
            error_data = resp.json()
            return None, error_data.get('error', 'unknown_error')

        # Other error
        else:
            return None, f"http_{resp.status_code}"

    except Exception as e:
        return None, "network_error"


def poll_and_wait_for_api_key(device_data: Dict[str, Any]) -> int:
    """
    Poll for API key until success or timeout.
    Returns exit code (0 = success, 1 = failure).
    """
    device_code = device_data['device_code']
    interval = device_data.get('interval', 5)
    expires_in = device_data.get('expires_in', 1800)
    start_time = _now_ts()
    poll_count = 0

    print("⏳ Waiting for authorization", end="", flush=True)

    while _now_ts() - start_time < expires_in:
        time.sleep(interval)

        creds, err = poll_for_api_key(device_code)

        # Got API key!
        if creds:
            save_credentials(creds)

            # Also update CLI config with the organization_id from auth
            if creds.get('organization_id'):
                set_active_org(creds['organization_id'])

            print("\n\n\033[1m✓ Login successful!\033[0m\n")
            return 0

        # Still waiting
        if err == 'authorization_pending':
            poll_count += 1
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            print(f"\r⏳ Waiting for authorization {spinner[poll_count % len(spinner)]}", end="", flush=True)
            continue

        # Slow down
        if err == 'slow_down':
            interval += 5
            continue

        # Access denied
        if err == 'access_denied':
            print("\n\n\033[91m✗ Access denied.\033[0m\n")
            return 1

        # Token expired
        if err in ('expired_token', 'invalid_grant'):
            print("\n\n\033[91m✗ Device code expired.\033[0m Run \033[1mreinforceenow login\033[0m again.\n")
            return 1

        # Unknown error
        print(f"\n\n\033[91m✗ Error: {err}\033[0m\n")
        return 1

    print("\n\n\033[91m✗ Login timed out.\033[0m Run \033[1mreinforceenow login\033[0m again.\n")
    return 1


def login_flow(wait: bool = True, force: bool = False) -> int:
    """
    Complete login flow: device authorization → poll → save API key.
    Returns exit code (0 = success, 1 = failure).
    """
    if not force and is_authenticated():
        print("Already logged in. Use \033[1mreinforceenow logout\033[0m or \033[1mreinforceenow login --force\033[0m to re-authenticate.")
        return 0

    # Start device flow
    device_data = begin_device_login()
    if not device_data:
        return 1

    # Poll for API key
    return poll_and_wait_for_api_key(device_data)
