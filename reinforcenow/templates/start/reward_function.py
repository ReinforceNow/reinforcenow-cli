# reward_function.py
# Simplified API: Individual reward functions combined with reward_aggregator

from reinforcenow import Sample, reward_function, reward_aggregator


@reward_function(name="accuracy")
async def check_accuracy(args, sample: Sample, **kwargs) -> float:
    """
    Reward function that checks if sentiment classification is correct.
    This will be tracked in the trace breakdown.

    Args:
        args: Additional arguments (unused in this example)
        sample: Sample containing prompt, response, and ground_truth
        **kwargs: Additional context

    Returns:
        Accuracy score: 1.0 for correct, 0.3 for valid format, 0.0 otherwise
    """
    response = sample.response.strip().lower()

    # Get ground_truth from metadata if available
    if hasattr(sample, "metadata") and sample.metadata:
        ground_truth = sample.metadata.get("ground_truth", "").lower()
    else:
        ground_truth = ""

    # Reward correct predictions
    if response == ground_truth:
        return 1.0
    elif response in ["positive", "negative", "neutral"]:
        return 0.3  # Partial credit for valid format
    else:
        return 0.0


@reward_function(name="format_quality")
async def check_format(args, sample: Sample, **kwargs) -> float:
    """
    Reward function that checks response format quality.
    This will be tracked in the trace breakdown.

    Returns:
        Format score: 1.0 if properly formatted, 0.5 otherwise
    """
    response = sample.response.strip()

    # Check if response is not empty and is one of the valid sentiments
    if response and response.lower() in ["positive", "negative", "neutral"]:
        return 1.0
    else:
        return 0.5


@reward_aggregator
async def reward(args, sample: Sample, **kwargs) -> float:
    """
    Main reward aggregator that combines individual reward functions.
    The breakdown is automatically tracked and logged in traces.

    Args:
        args: Additional arguments
        sample: Sample containing prompt, response, and ground_truth
        **kwargs: Additional context

    Returns:
        Total reward score
    """
    # Call individual reward functions - these are automatically tracked
    accuracy = await check_accuracy(args, sample, **kwargs)
    format_score = await check_format(args, sample, **kwargs)

    # Combine scores with weighting
    total_score = (accuracy * 0.8) + (format_score * 0.2)

    return total_score
