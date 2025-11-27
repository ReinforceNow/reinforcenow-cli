from rnow.core import reward, RewardArgs


@reward(parse_reasoning=True)
async def accuracy(args: RewardArgs, messages: list) -> float:
    """
    Simple accuracy reward for sentiment classification.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    response = messages[-1].get("content", "").strip().lower()
    ground_truth = str(args.metadata.get("ground_truth") or "").lower()

    # Simple exact match
    if response == ground_truth:
        return 1.0
    else:
        return 0.0