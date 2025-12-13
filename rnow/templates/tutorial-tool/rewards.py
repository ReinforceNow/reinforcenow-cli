from rnow.core import RewardArgs, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the boxed answer matches the expected answer."""
    pass
