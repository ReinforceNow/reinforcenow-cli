import re

import jellyfish

from rnow.core import RewardArgs, get_response, reward


@reward(precondition=True)
def used_browser(args: RewardArgs, messages: list) -> float:
    """Gate: must use browser tools to get any reward."""
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                if tc.get("function", {}).get("name", "").startswith("browser_"):
                    return 1.0
    return 0.0


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the final answer matches the expected answer."""
    response = get_response(messages)
    expected = args.metadata.get("expected_answer", "").strip().lower()

    match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if not match:
        return 0.0

    answer = match.group(1).strip().lower()
    similarity = jellyfish.jaro_winkler_similarity(answer, expected)
    return 1.0 if similarity > 0.9 else 0.0
