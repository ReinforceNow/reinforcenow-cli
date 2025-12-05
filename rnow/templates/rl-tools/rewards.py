from rnow.core import RewardArgs, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """
    Reward for finding the correct country using internet_search tool.
    """
    response = messages[-1].get("content", "")
    ground_truth = args.metadata.get("ground_truth")

    if ground_truth and ground_truth in response:
        return 1.0
    return 0.0
