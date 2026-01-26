"""
Computer Use Agent (CUA) Tools for VLM Browser Agent

Simple browser control tools inspired by OpenAI's computer-use-preview.
Each tool executes an action and returns a screenshot of the result.

Tools:
- click(x, y): Click at screen coordinates
- type_text(text): Type text into the focused element
- press_key(key): Press a keyboard key (Enter, Tab, etc.)
- scroll(direction, amount): Scroll the page
- wait(seconds): Wait for page to load
- navigate(url): Navigate to a URL

Unlike MCP/Playwright tools that return text snapshots, these tools
return screenshots for vision-language models to process.
"""

import asyncio
import base64
import os

from playwright.async_api import async_playwright, Page, Browser

from rnow.core import tool

# Global browser state (shared across tool calls in a rollout)
_browser: Browser | None = None
_page: Page | None = None
_playwright = None

# Display settings (match OpenAI CUA defaults)
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768


async def _get_page() -> Page:
    """Get or create the browser page."""
    global _browser, _page, _playwright

    if _page is None:
        # Get Browserbase CDP URL from environment
        browserbase_api_key = os.environ.get("BROWSERBASE_API_KEY")
        browserbase_session_id = os.environ.get("BROWSERBASE_SESSION_ID")

        _playwright = await async_playwright().start()

        if browserbase_api_key and browserbase_session_id:
            # Connect to Browserbase cloud browser
            cdp_url = f"wss://connect.browserbase.com?apiKey={browserbase_api_key}&sessionId={browserbase_session_id}"
            _browser = await _playwright.chromium.connect_over_cdp(cdp_url)
            contexts = _browser.contexts
            if contexts:
                _page = contexts[0].pages[0] if contexts[0].pages else await contexts[0].new_page()
            else:
                context = await _browser.new_context(
                    viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT}
                )
                _page = await context.new_page()
        else:
            # Local browser fallback
            _browser = await _playwright.chromium.launch(headless=True)
            _page = await _browser.new_page(
                viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT}
            )

        # Navigate to starting page
        await _page.goto("https://www.google.com")
        await asyncio.sleep(1)

    return _page


async def _screenshot_base64() -> str:
    """Take a screenshot and return as base64."""
    page = await _get_page()
    screenshot_bytes = await page.screenshot()
    return base64.b64encode(screenshot_bytes).decode("utf-8")


def _format_screenshot_response(action_desc: str, screenshot_b64: str) -> dict:
    """Format the response with action description and screenshot for VLM processing.

    Returns a special dict format that env.py recognizes as a VLM multimodal response.
    The __vlm_image__ key triggers multimodal message creation with the image embedded.
    """
    return {
        "__vlm_image__": {
            "data": screenshot_b64,
            "format": "png",  # Playwright screenshots are PNG by default
        },
        "text": action_desc,
    }


@tool
async def click(x: int, y: int) -> dict:
    """
    Click at the specified screen coordinates.

    The screen is 1024x768 pixels:
    - x=0 is the left edge, x=1023 is the right edge
    - y=0 is the top edge, y=767 is the bottom edge

    Use this to click on buttons, links, input fields, etc.
    After clicking an input field, use type_text() to enter text.

    Args:
        x: X coordinate (0-1023, left to right)
        y: Y coordinate (0-767, top to bottom)

    Returns:
        Description of the action and current page screenshot
    """
    page = await _get_page()

    # Clamp coordinates to valid range
    x = max(0, min(x, DISPLAY_WIDTH - 1))
    y = max(0, min(y, DISPLAY_HEIGHT - 1))

    await page.mouse.click(x, y)
    await asyncio.sleep(0.5)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Clicked at coordinates ({x}, {y})", screenshot)


@tool
async def type_text(text: str) -> dict:
    """
    Type text into the currently focused element.

    First use click() to focus an input field, then use this to type.
    The text is typed character by character.

    Args:
        text: The text to type

    Returns:
        Description of the action and current page screenshot
    """
    page = await _get_page()

    await page.keyboard.type(text)
    await asyncio.sleep(0.3)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Typed: \"{text}\"", screenshot)


@tool
async def press_key(key: str) -> dict:
    """
    Press a keyboard key.

    Common keys:
    - "Enter": Submit forms, confirm actions
    - "Tab": Move to the next form field
    - "Escape": Close dialogs, cancel actions
    - "Backspace": Delete the character before cursor
    - "ArrowUp", "ArrowDown": Navigate lists, scroll
    - "ArrowLeft", "ArrowRight": Move cursor in text

    Args:
        key: The key to press (e.g., "Enter", "Tab", "Escape")

    Returns:
        Description of the action and current page screenshot
    """
    page = await _get_page()

    await page.keyboard.press(key)
    await asyncio.sleep(0.3)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Pressed key: {key}", screenshot)


@tool
async def scroll(direction: str = "down", amount: int = 300) -> dict:
    """
    Scroll the page up or down.

    Use this to see content that's not currently visible.

    Args:
        direction: "up" or "down" (default: "down")
        amount: Pixels to scroll (default: 300, roughly one viewport section)

    Returns:
        Description of the action and current page screenshot
    """
    page = await _get_page()

    delta = -amount if direction.lower() == "up" else amount
    await page.evaluate(f"window.scrollBy(0, {delta})")
    await asyncio.sleep(0.3)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Scrolled {direction} by {amount} pixels", screenshot)


@tool
async def wait(seconds: float = 2.0) -> dict:
    """
    Wait for the page to update or load.

    Use this after actions that trigger page loads or dynamic content.

    Args:
        seconds: Time to wait in seconds (default: 2.0)

    Returns:
        Description of the action and current page screenshot
    """
    await asyncio.sleep(seconds)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Waited {seconds} seconds", screenshot)


@tool
async def navigate(url: str) -> dict:
    """
    Navigate to a specific URL.

    Use this to go directly to a website instead of searching.

    Args:
        url: The URL to navigate to (e.g., "https://www.google.com")

    Returns:
        Description of the action and current page screenshot
    """
    page = await _get_page()

    await page.goto(url)
    await asyncio.sleep(1)

    screenshot = await _screenshot_base64()
    return _format_screenshot_response(f"Navigated to: {url}", screenshot)


@tool
async def screenshot() -> dict:
    """
    Take a screenshot of the current page without performing any action.

    Use this to see the current state of the page.

    Returns:
        Current page screenshot
    """
    screenshot = await _screenshot_base64()
    return _format_screenshot_response("Current page state", screenshot)
