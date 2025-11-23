"""
Reward entry point for ReinforceNow.
"""
import inspect
from typing import Callable, Dict

# Global registry for reward functions
REWARD_REGISTRY: Dict[str, Callable] = {}


def reward(fn: Callable = None, *, description: str = None) -> Callable:
    """
    Decorator to register reward functions.

    Usage:
        @reward
        async def accuracy(args, sample):
            return 1.0

        @reward(description="Accuracy-based reward")
        async def accuracy(args, sample):
            return 1.0
    """
    def decorator(func):
        func._is_reward = True
        func._reward_name = func.__name__
        func._description = description or f"Reward: {func._reward_name}"

        # Register for introspection (secondary mechanism)
        REWARD_REGISTRY[func._reward_name] = func
        return func

    # Support both @reward and @reward(description="...")
    return decorator(fn) if fn else decorator