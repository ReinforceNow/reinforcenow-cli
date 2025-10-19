# reward_function.py
# Example reward functions using the new @reward decorator

from reinforcenow.core import reward


@reward
async def accuracy(args, sample, **kwargs):
    """
    Reward function that checks if sentiment classification is correct.

    Returns:
        Accuracy score: 1.0 for correct, 0.3 for valid format, 0.0 otherwise
    """
    # Handle different sample formats
    if isinstance(sample, dict):
        messages = sample.get("messages", [])
        if messages and len(messages) > 0:
            response = messages[-1].get("content", "").strip().lower()
        else:
            response = sample.get("response", "").strip().lower()

        metadata = sample.get("metadata", {})
        ground_truth = metadata.get("ground_truth", "").lower()
    else:
        # Handle object-style sample
        response = getattr(sample, "response", "").strip().lower()
        metadata = getattr(sample, "metadata", {})
        ground_truth = metadata.get("ground_truth", "").lower() if metadata else ""

    # Reward correct predictions
    if response == ground_truth:
        return 1.0
    elif response in ["positive", "negative", "neutral"]:
        return 0.3  # Partial credit for valid format
    else:
        return 0.0


@reward
async def format_quality(args, sample, **kwargs):
    """
    Reward function that checks response format quality.

    Returns:
        Format score: 1.0 if properly formatted, 0.5 otherwise
    """
    # Handle different sample formats
    if isinstance(sample, dict):
        messages = sample.get("messages", [])
        if messages and len(messages) > 0:
            response = messages[-1].get("content", "").strip()
        else:
            response = sample.get("response", "").strip()
    else:
        response = getattr(sample, "response", "").strip()

    # Check if response is not empty and is one of the valid sentiments
    if response and response.lower() in ["positive", "negative", "neutral"]:
        return 1.0
    else:
        return 0.5


@reward
async def combined_reward(args, sample, **kwargs):
    """
    Main reward that combines individual reward functions.

    Returns:
        Total reward score combining accuracy and format
    """
    # Call individual reward functions
    acc_score = await accuracy(args, sample, **kwargs)
    fmt_score = await format_quality(args, sample, **kwargs)

    # Combine scores with weighting
    total_score = (acc_score * 0.8) + (fmt_score * 0.2)

    return total_score