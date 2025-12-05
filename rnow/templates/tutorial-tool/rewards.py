from rnow.core import RewardArgs, reward


@reward
async def correctness(args: RewardArgs, messages: list) -> float:
    """
    Reward for finding the correct answer using internet_search tool.
    Uses exact match against the expected answer.
    """
    response = messages[-1].get("content", "").lower()
    answer = args.metadata.get("answer", "")

    if not answer:
        return 0.0

    # Check if answer appears in the response (case-insensitive)
    if answer.lower() in response:
        return 1.0

    return 0.0
