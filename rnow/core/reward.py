"""
Reward entry point for ReinforceNow.
"""
from typing import Callable, Dict

from rnow.models import RewardArgs

# Global registry for reward functions
REWARD_REGISTRY: Dict[str, Callable] = {}


def clear_reward_registry() -> None:
    """Clear the reward registry (useful for testing multiple projects)."""
    REWARD_REGISTRY.clear()


def reward(fn: Callable = None, *, description: str = None, parse_reasoning: bool = False) -> Callable:
    """
    Decorator to register reward functions.

    Usage:
        @reward
        async def accuracy(args: RewardArgs, messages: list) -> float:
            # args.metadata, args.variables contain data from train.jsonl
            # messages = conversation list
            ground_truth = args.metadata.get("ground_truth")
            return 1.0

        @reward(parse_reasoning=True)  # Auto-remove <think> tags from responses
        async def accuracy(args: RewardArgs, messages: list) -> float:
            return 1.0
    """
    import re

    def decorator(func):
        # Store metadata
        func._is_reward = True
        func._reward_name = func.__name__

        # If parse_reasoning is True, wrap the function to clean thinking tags
        if parse_reasoning:
            async def wrapper(args: RewardArgs, messages: list):
                # Clean the response by removing thinking tags
                if messages and messages[-1].get("role") == "assistant":
                    messages = messages.copy()
                    last_msg = messages[-1].copy()
                    content = last_msg.get("content", "")
                    # Remove <think>...</think> tags and surrounding whitespace
                    cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                    last_msg["content"] = cleaned
                    messages[-1] = last_msg

                return await func(args, messages)

            # Preserve function metadata
            wrapper._is_reward = True
            wrapper._reward_name = func.__name__

            # Register the WRAPPER in global registry, not the original function!
            REWARD_REGISTRY[func.__name__] = wrapper
            return wrapper
        else:
            # Register the original function if no wrapping needed
            REWARD_REGISTRY[func.__name__] = func
            return func

    # Support both @reward and @reward(description="...")
    return decorator(fn) if fn else decorator