"""Browser tools for VLM agent training via Kernel cloud browsers.

Uses Qwen3-VL coordinate format: coordinate=[x, y] with 0-1000 scale.

Browser is acquired lazily on first tool call from the pool specified
by KERNEL_POOL_NAME environment variable.

All tools use sandbox=True to run inside the Docker container.
"""

import contextlib
import os

import requests

from rnow.core import tool

KERNEL_URL = "https://api.onkernel.com"
WIDTH, HEIGHT = 1024, 768

# Cache for session ID (acquired lazily)
_session_id_cache = None


def _get_session_id() -> str:
    """Get or acquire a Kernel browser session."""
    global _session_id_cache

    # Return cached session if available
    if _session_id_cache:
        return _session_id_cache

    # Check environment variable
    session_id = os.environ.get("KERNEL_SESSION_ID")
    if session_id:
        _session_id_cache = session_id
        return session_id

    # Try reading from file
    try:
        with open("/tmp/kernel_session_id") as f:
            session_id = f.read().strip()
            if session_id:
                _session_id_cache = session_id
                return session_id
    except FileNotFoundError:
        pass

    # Acquire browser from pool
    session_id = _acquire_browser()
    _session_id_cache = session_id
    return session_id


def _acquire_browser() -> str:
    """Acquire a browser from the pool."""
    api_key = os.environ.get("KERNEL_API_KEY")
    if not api_key:
        raise RuntimeError("KERNEL_API_KEY environment variable not set")

    pool_name = os.environ.get("KERNEL_POOL_NAME", "rnow-browsers")
    acquire_timeout = int(os.environ.get("KERNEL_ACQUIRE_TIMEOUT", "300"))

    print(f"[Kernel] Acquiring browser from pool {pool_name}...", flush=True)

    resp = requests.post(
        f"{KERNEL_URL}/browser_pools/{pool_name}/acquire",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"acquire_timeout_seconds": acquire_timeout},
        timeout=acquire_timeout + 10,
    )

    if not resp.ok:
        raise RuntimeError(f"Failed to acquire browser: {resp.status_code} {resp.text[:500]}")

    data = resp.json()
    session_id = data.get("session_id")
    if not session_id:
        raise RuntimeError(f"No session_id in response: {data}")

    # Save to file for cleanup
    with open("/tmp/kernel_session_id", "w") as f:
        f.write(session_id)

    print(f"[Kernel] Browser acquired: {session_id}", flush=True)

    # Start replay recording (optional, ignore errors)
    with contextlib.suppress(Exception):
        requests.post(
            f"{KERNEL_URL}/browsers/{session_id}/replays",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )

    return session_id


def _api(method: str, path: str, json: dict = None) -> requests.Response:
    """Make an API call to Kernel."""
    api_key = os.environ.get("KERNEL_API_KEY")
    if not api_key:
        raise RuntimeError("KERNEL_API_KEY environment variable not set")

    resp = requests.request(
        method,
        f"{KERNEL_URL}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        json=json,
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Kernel API error {resp.status_code}: {resp.text[:500]}")
    return resp


def _run(code: str) -> dict:
    """Execute Playwright code on the browser."""
    session_id = _get_session_id()
    print(f"[Kernel] Executing: {code[:100]}...", flush=True)
    result = _api("POST", f"/browsers/{session_id}/playwright/execute", {"code": code}).json()
    print(
        f"[Kernel] Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}",
        flush=True,
    )
    return result


def _screenshot() -> dict:
    """Take a screenshot and return in VLM format."""
    response = _run('const b = await page.screenshot(); return b.toString("base64")')
    b64 = response.get("result", "")

    # Debug: log screenshot result
    if not b64:
        print(f"[Kernel] Screenshot returned empty. Full response: {response}", flush=True)
        return {
            "__vlm_image__": {"data": "", "format": "png"},
            "text": "Screenshot failed - empty response",
        }

    print(f"[Kernel] Screenshot captured: {len(b64)} bytes base64", flush=True)
    return {"__vlm_image__": {"data": b64, "format": "png"}}


def _to_pixels(coord: list[int]) -> tuple[int, int]:
    """Convert 0-1000 coordinate to pixels."""
    x = int(coord[0] / 1000 * WIDTH)
    y = int(coord[1] / 1000 * HEIGHT)
    return max(0, min(x, WIDTH - 1)), max(0, min(y, HEIGHT - 1))


@tool(sandbox=True)
def click(coordinate: list[int]) -> dict:
    """Click at [x, y] coordinate (0-1000 scale). Example: [500, 300] clicks center-top."""
    x, y = _to_pixels(coordinate)
    _run(f"await page.mouse.click({x}, {y})")
    return _screenshot()


@tool(sandbox=True)
def type_text(text: str) -> dict:
    """Type text into focused element."""
    _run(f"await page.keyboard.type({repr(text)})")
    return _screenshot()


@tool(sandbox=True)
def press_key(key: str) -> dict:
    """Press key (Enter, Tab, Escape, Backspace, ArrowUp, ArrowDown)."""
    _run(f"await page.keyboard.press('{key}')")
    return _screenshot()


@tool(sandbox=True)
def scroll(coordinate: list[int], direction: str = "down") -> dict:
    """Scroll at [x, y] coordinate (0-1000 scale), direction up/down."""
    x, y = _to_pixels(coordinate)
    delta = -300 if direction == "up" else 300
    _run(f"await page.mouse.move({x}, {y})")
    _run(f"await page.mouse.wheel(0, {delta})")
    return _screenshot()


@tool(sandbox=True)
def navigate(url: str) -> dict:
    """Navigate to URL."""
    _run(f"await page.goto('{url}')")
    return _screenshot()


@tool(sandbox=True)
def screenshot() -> dict:
    """Get current page screenshot."""
    return _screenshot()
