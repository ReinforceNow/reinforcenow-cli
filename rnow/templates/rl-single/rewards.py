import re

from rnow.core import RewardArgs, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the boxed answer matches the expected answer."""
    response = messages[-1]["content"]
    expected_raw = args.metadata["expected_answer"].strip()

    # Split expected answers like "A or B"
    expected = {s.strip() for s in expected_raw.split(" or ")}

    # Extract ALL boxed answers (handles \boxed and \\boxed)
    answers = {s.strip() for s in re.findall(r"\\+boxed\{([^}]*)\}", response)}

    if not answers:
        return 0.0

    # Success if ANY intersection between expected answers and boxed answers
    return 1.0 if expected & answers else 0.0
