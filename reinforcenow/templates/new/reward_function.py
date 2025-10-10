# reward_function.py
# Define your reward function for RL training

def reward(prompt: str, response: str, **kwargs) -> float:
    """
    Calculate reward score for a generated response.

    Args:
        prompt: Input prompt
        response: Generated response
        **kwargs: Additional context (ground truth, metadata, etc.)

    Returns:
        Reward score (higher is better)
    """
    # Example: Simple length-based reward
    # Replace with your actual reward logic

    if len(response) > 10:
        return 1.0
    return 0.0
