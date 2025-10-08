# auth.py
# Non-blocking device login helpers for ReinforceNow CLI

import base64
import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load .env from current working directory (optional - has defaults)
load_dotenv()

# === Configuration (with sane production defaults; overridable via .env) ===
BASE_URL = os.getenv("REINMAX_BASE_URL", "https://www.reinforcenow.ai").rstrip("/")
CLIENT_ID = os.getenv("REINMAX_CLIENT_ID", "better-auth-cli")
USER_AGENT = os.getenv("REINMAX_USER_AGENT", "Reinmax-CLI/1.1")

DEVICE_AUTH_URL = os.getenv("REINMAX_DEVICE_AUTH_URL", f"{BASE_URL}/api/auth/device/code")
TOKEN_URL = os.getenv("REINMAX_TOKEN_URL", f"{BASE_URL}/api/auth/device/token")
SESSION_URL = os.getenv("REINMAX_SESSION_URL", f"{BASE_URL}/api/auth/get-session")

# Files under ~/.reinmax
TOKEN_DIR = Path.home() / ".reinmax"
TOKEN_FILE = TOKEN_DIR / "token.json"
PENDING_FILE = TOKEN_DIR / "pending_device.json"


# === Utilities ===
def _ensure_dirs() -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)


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


def _decode_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    """Best-effort decode of a JWT payload without verification (for exp inspection)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        # Add base64 padding if needed
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return None


# === Public helpers ===
def is_authenticated() -> bool:
    """Fast check using local token file + lightweight remote validation if possible."""
    return validate_token()


def validate_token() -> bool:
    """
    Validate token on disk. If it's structurally valid and not obviously expired, we
    try a lightweight remote call. If remote validation fails with 401, we remove it.
    """
    if not TOKEN_FILE.exists():
        return False

    try:
        data = _load_json(TOKEN_FILE) or {}
        access_token = data.get("access_token")
        if not access_token:
            TOKEN_FILE.unlink(missing_ok=True)
            return False

        # If token has exp and it's in the past, treat as invalid (no guesswork).
        payload = _decode_jwt_payload(access_token) or {}
        exp = payload.get("exp")
        if isinstance(exp, (int, float)) and exp <= _now_ts():
            # Try refresh if we have a refresh token; otherwise remove.
            if not maybe_refresh_token():
                TOKEN_FILE.unlink(missing_ok=True)
                return False
            # Re-read after refresh
            data = _load_json(TOKEN_FILE) or {}
            access_token = data.get("access_token")
            if not access_token:
                return False

        # Remote validation (best-effort). Some backends accept bearer for get-session.
        headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
        try:
            resp = requests.get(SESSION_URL, headers=headers, timeout=10)
            if resp.status_code == 200:
                return True
            if resp.status_code == 401:
                TOKEN_FILE.unlink(missing_ok=True)
                return False
            # Non-401: server hiccup → assume token might still be okay.
            return True
        except requests.RequestException:
            # Network issue → assume token *might* be okay so we don't thrash.
            return True

    except Exception:
        TOKEN_FILE.unlink(missing_ok=True)
        return False


def maybe_refresh_token() -> bool:
    """
    Attempt refresh if refresh_token is present.
    Returns True if we refreshed, False otherwise.
    """
    data = _load_json(TOKEN_FILE) or {}
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        return False

    try:
        tok = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": USER_AGENT,
            },
            timeout=30,
        )
        if tok.status_code == 200:
            token_data = tok.json()
            _save_json(TOKEN_FILE, token_data)
            return True
        # If refresh fails, remove the old token to force a clean login next time.
        TOKEN_FILE.unlink(missing_ok=True)
        return False
    except requests.RequestException:
        # Network error → keep existing token (we'll try again later).
        return True
    except Exception:
        TOKEN_FILE.unlink(missing_ok=True)
        return False


def get_auth_headers() -> Dict[str, str]:
    """
    Return Authorization headers or raise a clear RuntimeError if not authenticated.
    Callers that want graceful behavior can check is_authenticated() first.
    """
    if not TOKEN_FILE.exists():
        raise RuntimeError("Not authenticated. Run `reinforcenow login`.")
    data = _load_json(TOKEN_FILE) or {}
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Not authenticated. Run `reinforcenow login`.")
    return {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}


# === Device Flow (non-blocking by default) ===
def begin_device_login() -> Optional[Dict[str, Any]]:
    """
    Start device authorization, open the browser, persist pending context,
    and RETURN immediately. Caller can choose to poll later with finish_device_login().
    """
    try:
        resp = requests.post(
            DEVICE_AUTH_URL,
            data={"client_id": CLIENT_ID, "scope": "openid profile email offline_access"},
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Unable to reach auth server: {e}")
        return None

    if resp.status_code != 200:
        print(f"Device authorization failed: {resp.text}")
        return None

    payload = resp.json()
    device_code = payload["device_code"]
    user_code = payload["user_code"]
    verification_uri = payload["verification_uri"]
    verification_uri_complete = payload.get("verification_uri_complete")
    interval = int(payload.get("interval", 5))
    expires_in = int(payload.get("expires_in", 600))
    expires_at = _now_ts() + expires_in

    # Persist pending context so user can finish later without re-issuing a code.
    pending = {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "interval": interval,
        "expires_at": expires_at,
    }
    _save_json(PENDING_FILE, pending)

    # Build device URL with user_code
    device_url = f"{verification_uri}?user_code={user_code}"

    # Friendly output
    print(f"\nOpening authorization page in your browser...")
    print(f"URL: {device_url}")
    print(f"Authorization code: \033[1m{user_code}\033[0m\n")

    # Try to open the browser immediately.
    try:
        webbrowser.open(device_url)
    except Exception:
        pass

    # Give browser a moment to open
    time.sleep(1)

    # Clear instructions for the user
    print("\033[1mComplete the authorization in your browser to continue.\033[0m")
    print("\033[91mLogin first if you have to.\033[0m")
    print()

    return pending


def _poll_for_token_once(device_code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Single poll attempt. Returns (token_data, error_code). error_code may be:
    'authorization_pending', 'slow_down', 'access_denied', 'expired_token', 'invalid_grant', or None.
    """
    tok = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT},
        timeout=30,
    )

    if tok.status_code == 200:
        return tok.json(), None

    try:
        err_data = tok.json()
        return None, err_data.get("error")
    except Exception:
        return None, "unknown_error"


def finish_device_login(wait: bool = True) -> int:
    """
    If there is a pending device login on disk, poll for the token.
    Returns a process exit code (0 success, 1 failure).
    """
    pending = _load_json(PENDING_FILE)
    if not pending:
        print("No pending login found. Run `reinforcenow login` to start again.")
        return 1

    device_code = pending["device_code"]
    # Use 1 second polling for faster response (ignore server's suggestion)
    interval = 1
    expires_at = int(pending.get("expires_at", _now_ts()))

    # If not waiting, do a single probe and exit early (non-blocking).
    if not wait:
        token, err = _poll_for_token_once(device_code)
        if token:
            _save_json(TOKEN_FILE, token)
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            print("\033[1mLogin successful.\033[0m\n")
            return 0

        # Not authorized yet → exit cleanly, tell user to finish in browser.
        if err in ("authorization_pending", "slow_down"):
            print("Waiting for authorization. Re-run with \033[1m--wait\033[0m to finish here.")
            return 1

        if err == "access_denied":
            print("Access denied.")
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return 1

        if err in ("expired_token", "invalid_grant"):
            print("Device code expired. Run \033[1mreinforceenow login\033[0m to start again.")
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return 1

        print("Unexpected response. Try again with \033[1m--wait\033[0m or restart login.")
        return 1

    # Blocking behavior: poll until success/error/expiry with progress indicators
    start_time = _now_ts()
    last_reminder = start_time
    poll_count = 0

    print("⏳ Waiting for authorization", end="", flush=True)

    while _now_ts() < expires_at:
        token, err = _poll_for_token_once(device_code)
        if token:
            _save_json(TOKEN_FILE, token)
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            print("\n\n\033[1m✓ Login successful.\033[0m\n")
            return 0

        if err == "authorization_pending":
            poll_count += 1
            current_time = _now_ts()
            elapsed = current_time - start_time

            # Show spinner
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            print(f"\r⏳ Waiting for authorization {spinner[poll_count % len(spinner)]}", end="", flush=True)

            # Show helpful reminders at intervals
            if current_time - last_reminder >= 30:  # Every 30 seconds
                user_code = pending.get("user_code", "")
                verification_uri = pending.get("verification_uri", BASE_URL + "/device")
                device_url = f"{verification_uri}?user_code={user_code}"

                if elapsed >= 120:  # After 2 minutes
                    print(f"\n\n\033[1m⚠ Still waiting... Make sure you've:\033[0m")
                    print(f"  1. \033[91mLogged into the website:\033[0m {BASE_URL}/login")
                    print(f"  2. \033[91mOpened the authorization page:\033[0m {device_url}")
                    print(f"  3. \033[91mClicked the\033[0m \033[1mApprove\033[0m \033[91mbutton\033[0m")
                    print(f"\n⏳ Waiting for authorization", end="", flush=True)
                elif elapsed >= 60:  # After 1 minute
                    print(f"\n\n\033[1mℹ If the browser didn't open, visit:\033[0m {device_url}")
                    print(f"\n⏳ Waiting for authorization", end="", flush=True)

                last_reminder = current_time

            time.sleep(interval)
            continue

        if err == "slow_down":
            interval += 5
            time.sleep(interval)
            continue

        if err == "access_denied":
            print("\n\n\033[91m✗ Access denied.\033[0m\n")
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return 1

        if err in ("expired_token", "invalid_grant"):
            print("\n\n\033[91m✗ Device code expired.\033[0m Run \033[1mreinforceenow login\033[0m to start again.\n")
            try:
                PENDING_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return 1

        # Unknown hiccup; wait a bit and retry to be resilient.
        time.sleep(max(2, interval))

    print("Login timed out. Run \033[1mreinforceenow login\033[0m to start again.")
    try:
        PENDING_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return 1


def login_flow(wait: bool = True, force: bool = False) -> int:
    """
    High-level login interface used by the CLI command.
    - If already authenticated and not forcing: print status and return 0.
    - Otherwise, start device login and wait until finished (default)
      or return immediately if wait=False.
    """
    if not force and is_authenticated():
        print("Already logged in. Use \033[1mreinforceenow logout\033[0m or \033[1mreinforceenow login --force\033[0m to re-authenticate.")
        return 0

    # Start (or resume) device flow
    started = begin_device_login()
    if not started:
        return 1

    # If the caller wants to wait, finish now; else return immediately.
    if wait:
        print("Waiting for authorization...")
        return finish_device_login(wait=True)

    print("Run \033[1mreinforceenow login\033[0m (without --no-wait) to complete.")
    return 0
