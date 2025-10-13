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


def reward_function(func: Callable) -> Callable:
    """
    Decorator to mark a function as a reward function.

    Usage:
        @reward_function
        async def code_brackets(args, sample: Sample, **kwargs) -> float:
            return 1.0 if check_passed else 0.0
    """
    func.__reward_function__ = True
    return func


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
