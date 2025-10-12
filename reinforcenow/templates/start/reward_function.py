# reward_function.py
# Quickstart Tutorial: Reward Function for Sentiment Analysis

from reinforcenow import Sample, reward_function, reward_executor


# Simple reward function using decorator
@reward_function
async def sentiment_accuracy(args, sample: Sample, **kwargs) -> float:
    """
    Reward function that checks if sentiment classification is correct.

    Args:
        args: Additional arguments (unused in this example)
        sample: Sample containing prompt, response, and ground_truth
        **kwargs: Additional context

    Returns:
        Reward score: 1.0 for correct, 0.0 for incorrect
    """
    response = sample["response"].strip().lower()
    ground_truth = sample.get("ground_truth", "").lower()

    # Reward correct predictions
    if response == ground_truth:
        return 1.0
    elif response in ["positive", "negative", "neutral"]:
        return 0.3  # Partial credit for valid format
    else:
        return 0.0


# Optional: Custom executor for more complex reward logic
@reward_executor
async def custom_executor(args, sample: Sample, **kwargs) -> dict:
    """
    Custom reward executor that combines multiple reward functions.

    Args:
        args: Additional arguments
        sample: Sample containing prompt, response, and ground_truth
        **kwargs: Additional context

    Returns:
        dict with 'score' (total reward) and 'pred' (breakdown)
    """
    # Run the sentiment accuracy check
    accuracy_score = await sentiment_accuracy(args, sample, **kwargs)

    # Could add more reward functions here
    # format_score = await check_format(args, sample, **kwargs)
    # length_score = await check_length(args, sample, **kwargs)

    # Combine scores
    total_score = accuracy_score

    breakdown = {
        "accuracy": accuracy_score,
        # "format": format_score,
        # "length": length_score,
    }

    return {
        "score": total_score,
        "pred": breakdown
    }


# Legacy API support (optional, for backwards compatibility)
def reward(prompt: str, response: str, **kwargs) -> float:
    """
    Legacy reward function for backwards compatibility.

    This is automatically converted to the new async format.
    """
    ground_truth = kwargs.get("ground_truth", "").lower()
    response = response.strip().lower()

    if response == ground_truth:
        return 1.0
    elif response in ["positive", "negative", "neutral"]:
        return 0.3
    else:
        return 0.0
