from rnow.core import reward


@reward(parse_reasoning=True)
async def accuracy(args, sample, **kwargs):
    """
    Reward for finding the correct country using internet_search tool.
    """
    messages = sample.get("messages", [])
    response = messages[-1].get("content", "").strip().lower()
    ground_truth = sample.get("metadata", {}).get("ground_truth_country", "").lower()

    if ground_truth in response:
        return 1.0
    return 0.0
