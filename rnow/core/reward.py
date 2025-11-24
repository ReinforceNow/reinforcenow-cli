"""
Reward entry point for ReinforceNow.
"""
import inspect
from typing import Callable, Dict

# Global registry for reward functions
REWARD_REGISTRY: Dict[str, Callable] = {}


def reward(fn: Callable = None, *, description: str = None, parse_reasoning: bool = False) -> Callable:
    """
    Decorator to register reward functions.

    Usage:
        @reward
        async def accuracy(args, sample):
            return 1.0

        @reward(description="Accuracy-based reward")
        async def accuracy(args, sample):
            return 1.0

        @reward(parse_reasoning=True)  # Auto-remove <think> tags from responses
        async def accuracy(args, sample):
            return 1.0
    """
    import re

    def decorator(func):
        # Store metadata
        func._is_reward = True
        func._reward_name = func.__name__

        # If parse_reasoning is True, wrap the function to clean thinking tags
        if parse_reasoning:
            async def wrapper(args, sample, **kwargs):
                # Clean the response by removing thinking tags
                if sample and "messages" in sample:
                    messages = sample.get("messages", [])
                    if messages and messages[-1].get("role") == "assistant":
                        # Create a modified sample with cleaned content
                        sample = sample.copy()
                        messages = messages.copy()
                        last_msg = messages[-1].copy()
                        content = last_msg.get("content", "")
                        # Remove <think>...</think> tags and surrounding whitespace
                        cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                        last_msg["content"] = cleaned
                        messages[-1] = last_msg
                        sample["messages"] = messages

                return await func(args, sample, **kwargs)

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