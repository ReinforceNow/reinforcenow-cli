# reward_function.py
# Define your reward function for RL training

from reinforcenow import Sample, reward_function, reward_executor


# Example reward function using the new decorator pattern
@reward_function
async def example_reward(args, sample: Sample, **kwargs) -> float:
    """
    Calculate reward score for a generated response.

    Args:
        args: Additional arguments
        sample: Sample containing prompt, response, and optional ground_truth
        **kwargs: Additional context

    Returns:
        Reward score: 1.0 for pass, 0.0 for fail
    """
    response = sample.response

    # Example: Simple length-based reward
    # Replace with your actual reward logic
    if len(response) > 10:
        return 1.0
    return 0.0


# Optional: Define custom executor for complex reward logic
# @reward_executor
# async def custom_executor(args, sample: Sample, **kwargs) -> dict:
#     """
#     Custom reward executor that can combine multiple reward functions.
#
#     Returns:
#         dict with 'score' (total reward) and 'pred' (breakdown)
#     """
#     score1 = await example_reward(args, sample, **kwargs)
#     # score2 = await another_reward(args, sample, **kwargs)
#
#     return {
#         "score": score1,
#         "pred": {"example": score1}
#     }

