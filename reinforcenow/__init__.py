"""
ReinforceNow CLI - Command-line interface for ReinforceNow RLHF platform.
"""

# Export types for user convenience
from reinforcenow.types import (
    Sample,
    RewardFunction,
    RewardExecutor,
    GenerationFunction,
    reward_function,
    reward_aggregator,
    reward_executor,
    generation,
)

__all__ = [
    "Sample",
    "RewardFunction",
    "RewardExecutor",
    "GenerationFunction",
    "reward_function",
    "reward_aggregator",
    "reward_executor",
    "generation",
]

__version__ = "0.8.2"
