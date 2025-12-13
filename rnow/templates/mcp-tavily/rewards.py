import re

import jellyfish

from rnow.core import RewardArgs, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the final answer matches the expected answer."""
    response = messages[-1].get("content", "")
    expected = args.metadata.get("expected_answer", "").strip().lower()

    # Extract content after "Final Answer:"
    match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if not match:
        return 0.0

    answer = match.group(1).strip().lower()

    # Use Jaro-Winkler similarity (1.0 = exact match, 0.0 = no similarity)
    similarity = jellyfish.jaro_winkler_similarity(answer, expected)

    # Require high similarity (>0.9) to count as correct
    return 1.0 if similarity > 0.9 else 0.0
