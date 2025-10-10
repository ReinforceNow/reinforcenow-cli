# reward_function.py
# Quickstart Tutorial: Reward Function for Sentiment Analysis

def reward(prompt: str, response: str, **kwargs) -> float:
    """
    Calculate reward score for sentiment analysis predictions.

    This quickstart example rewards correct sentiment classifications.

    Args:
        prompt: Input text containing the review and ground truth label
        response: Generated sentiment classification
        **kwargs: Additional context (can include ground_truth if provided)

    Returns:
        Reward score: 1.0 for correct, -0.5 for incorrect
    """
    # Extract ground truth from kwargs if provided
    ground_truth = kwargs.get("ground_truth", "").lower()
    response = response.strip().lower()

    # Dummy implementation for quickstart tutorial
    # In practice, you'd have ground truth labels in your dataset

    if not ground_truth:
        # If no ground truth, use simple heuristics for demonstration
        # In real usage, always provide ground truth labels in your data
        if response in ["positive", "negative", "neutral"]:
            return 0.5  # Valid format
        return -0.5  # Invalid format

    # Reward correct predictions
    if response == ground_truth:
        return 1.0
    else:
        return -0.5
