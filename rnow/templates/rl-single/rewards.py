from rnow.core import reward


@reward(parse_reasoning=True)
async def accuracy(args, sample, **kwargs):
    """
    Simple accuracy reward for sentiment classification.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    # Get the response from messages
    messages = sample.get("messages", [])

    response = messages[-1].get("content", "").strip().lower()
    ground_truth = sample.get("metadata", {}).get("ground_truth", "").lower()

    # Simple exact match
    if response == ground_truth:
        return 1.0
    else:
        return 0.0