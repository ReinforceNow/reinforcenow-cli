"""
ReinforceNow Core - Entry points for reward and tool decorators.

Users only need:
- @reward decorator for defining reward functions
- @tool decorator for defining tool functions
- RewardArgs for type hints in reward functions
"""

from rnow.models import RewardArgs

from .reward import (
    REWARD_REGISTRY,
    clear_reward_registry,
    compute_total_reward,
    is_precondition,
    reward,
)
from .tool import TOOL_REGISTRY, clear_tool_registry, tool

__all__ = [
    # User-facing API
    "reward",
    "tool",
    "RewardArgs",
    # Registries (used by CLI and trainer)
    "REWARD_REGISTRY",
    "TOOL_REGISTRY",
    "clear_reward_registry",
    "clear_tool_registry",
    "is_precondition",
    "compute_total_reward",
]
