"""
Type definitions for the reward and generation systems.

Users decorate their functions with @reward_function or @generation to mark them
for automatic detection and execution.
"""
from typing import TypedDict, Protocol, Callable, Awaitable, Any, Optional


class Sample(TypedDict, total=False):
    """
    Sample data structure for reinforcement learning.

    Users work with these fields:
    """
    # User-facing fields
    prompt: str | list[dict[str, str]]  # Input prompt (text or multimodal)
    response: str                        # Generated response
    metadata: dict                       # User metadata

    # Slime adds these automatically (users can read but shouldn't set):
    tokens: list[int]                   # Tokenized representation
    response_length: int                # Length of response
    reward: float | dict[str, Any]      # Computed reward
    status: Any                         # Generation status


class RewardFunction(Protocol):
    """
    Protocol for reward functions.

    Reward functions must:
    - Be async functions
    - Accept (args, sample: Sample, **kwargs) OR (args, code: str, **kwargs)
    - Return float (1.0 for pass, 0.0 for fail)
    - Have __reward_function__ = True attribute
    """
    __reward_function__: bool
    __name__: str

    def __call__(
        self,
        args,
        input: Sample | str,
        **kwargs
    ) -> Awaitable[float]:
        ...


class RewardExecutor(Protocol):
    """
    Protocol for reward executor functions.

    The executor is the entry point called by slime.
    It must:
    - Be an async function
    - Accept (args, sample: Sample, **kwargs)
    - Return dict with 'score' and 'pred'
    - Have __reward_executor__ = True attribute
    """
    __reward_executor__: bool
    __name__: str

    def __call__(
        self,
        args,
        sample: Sample,
        **kwargs
    ) -> Awaitable[dict]:
        ...


def reward_function(func: Callable = None, *, name: str = None) -> Callable:
    """
    Decorator to mark a function as a reward function.

    Individual reward functions can be called standalone or combined
    in a @reward_aggregator for automatic tracking.

    Args:
        name: Name for this reward in breakdowns (defaults to function name)

    Usage:
        @reward_function
        async def simple_reward(args, sample: Sample, **kwargs) -> float:
            return 1.0 if check_passed else 0.0

        @reward_function(name="accuracy")
        async def check_accuracy(args, sample: Sample, **kwargs) -> float:
            return 1.0 if correct else 0.0
    """
    def decorator(f: Callable) -> Callable:
        f.__reward_function__ = True
        f.__reward_name__ = name or f.__name__
        return f

    if func is None:
        # Called with arguments: @reward_function(name="accuracy")
        return decorator
    else:
        # Called without arguments: @reward_function
        return decorator(func)


def reward_executor(func: Callable) -> Callable:
    """
    Decorator to mark a function as the reward executor.

    Usage:
        @reward_executor
        async def custom_executor(args, sample: Sample, **kwargs) -> dict:
            # Your custom execution logic
            return {"score": total_reward, "pred": breakdown}

    If not defined by user, a default executor is automatically provided.
    """
    func.__reward_executor__ = True
    return func


class GenerationFunction(Protocol):
    """
    Protocol for generation functions.

    Generation functions must:
    - Be async functions
    - Accept (args, sample: Sample, sampling_params: dict, state: Any, **kwargs)
    - Return Sample (updated with response, tokens, and status)
    - Have __generation__ = True attribute
    """
    __generation__: bool
    __name__: str

    def __call__(
        self,
        args,
        sample: Sample,
        sampling_params: dict[str, Any],
        state: Optional[Any] = None,
        **kwargs
    ) -> Awaitable[Sample]:
        ...


def reward_aggregator(func: Callable = None, *, breakdown: bool = True) -> Callable:
    """
    Decorator to mark a function as a reward aggregator.

    Reward aggregators combine multiple @reward_function calls and their scores
    are automatically tracked and logged in the trace breakdown.

    Args:
        breakdown: If True, automatically tracks reward_function calls (default: True)

    Usage:
        @reward_function(name="accuracy")
        async def check_accuracy(args, sample: Sample, **kwargs) -> float:
            return 1.0 if correct else 0.0

        @reward_function(name="format")
        async def check_format(args, sample: Sample, **kwargs) -> float:
            return 1.0 if valid_format else 0.0

        @reward_aggregator
        async def combined_reward(args, sample: Sample, **kwargs) -> float:
            # These calls are automatically tracked in the breakdown
            accuracy = await check_accuracy(args, sample)
            format_score = await check_format(args, sample)
            return (accuracy * 0.8) + (format_score * 0.2)
    """
    def decorator(f: Callable) -> Callable:
        f.__reward_aggregator__ = True
        f.__reward_breakdown__ = breakdown
        return f

    if func is None:
        # Called with arguments: @reward_aggregator(breakdown=True)
        return decorator
    else:
        # Called without arguments: @reward_aggregator
        return decorator(func)


def generation(func: Callable) -> Callable:
    """
    Decorator to mark a function as a generation function.

    Usage:
        @generation
        async def custom_generate(args, sample: Sample, sampling_params, state) -> Sample:
            # Your custom generation logic
            # Update sample with response, tokens, status
            return sample
    """
    func.__generation__ = True
    return func
