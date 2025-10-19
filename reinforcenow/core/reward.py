"""
Reward entry point for ReinforceNow.
"""
import inspect
from typing import Callable, Dict

# Global registry for reward functions
REWARD_REGISTRY: Dict[str, Callable] = {}


def reward(fn: Callable = None, *, description: str = None) -> Callable:
    """
    Decorator to register reward functions using the function name.

    Usage:
        @reward
        async def accuracy(args, sample):
            return 1.0
    """
    def decorator(func):
        # Register with function name as key
        REWARD_REGISTRY[func.__name__] = func

        # Add metadata
        func._is_reward = True
        func._description = description or f"Reward function: {func.__name__}"

        return func

    # Support both @reward and @reward(description="...")
    if fn is None:
        return decorator
    return decorator(fn)