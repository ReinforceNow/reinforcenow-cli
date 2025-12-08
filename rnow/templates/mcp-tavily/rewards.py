import re

from rnow.core import RewardArgs, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the boxed answer matches the expected answer."""
    response = messages[-1].get("content", "")
    expected = args.metadata.get("expected_answer", "")

    # Extract content from \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", response, re.DOTALL)
    if not match:
        return 0.0

    answer = match.group(1).strip()

    return 1.0 if expected in answer or answer in expected else 0.0
