"""Reward functions for browser-based QA."""

import re

from rnow.core import RewardArgs, get_response, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the final answer matches the expected answer.

    Looks for "Final Answer: <answer>" in the response and compares
    it to the expected answer using fuzzy string matching.

    Returns:
        1.0 if the answer is correct (>90% similarity), 0.0 otherwise
    """
    # Lazy import to avoid loading in sidecar unnecessarily
    import jellyfish

    response = get_response(messages)
    expected = args.metadata.get("expected_answer", "").strip().lower()

    if not expected:
        return 0.0

    # Extract content after "Final Answer:"
    match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if not match:
        return 0.0

    answer = match.group(1).strip().lower()

    # Use Jaro-Winkler similarity (1.0 = exact match, 0.0 = no similarity)
    similarity = jellyfish.jaro_winkler_similarity(answer, expected)

    # Require high similarity (>0.9) to count as correct
    return 1.0 if similarity > 0.9 else 0.0


@reward
def used_browse(args: RewardArgs, messages: list) -> float:
    """Reward for using browser tools to gather information.

    Encourages the model to actually browse web pages rather than
    guessing answers without research. Works with Playwright MCP tools.

    Returns:
        0.2 if any browser tool was used, 0.0 otherwise
    """
    browser_tools = {
        "browser_navigate",
        "browser_click",
        "browser_snapshot",
        "browser_type",
        "browser_scroll",
        "browser_go_back",
        "browse",
        "navigate",
        "click",
        "type",
        "snapshot",
    }
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                if func_name in browser_tools or func_name.startswith("browser_"):
                    return 0.2
    return 0.0
