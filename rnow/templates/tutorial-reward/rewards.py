from rnow.core import reward, RewardArgs


@reward(parse_reasoning=True)
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Simple accuracy: 1.0 if answer matches ground_truth, 0.0 otherwise."""

    # Complete the reward
