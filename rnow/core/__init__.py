"""
rnow core - Entry points for reward and tool decorators.

Users only need:
- @reward decorator for defining reward functions
- @tool decorator for defining tool functions
- RewardArgs for type hints in reward functions
- llm_judge for LLM-as-a-judge evaluation
"""

from rnow.models import RewardArgs, ToolArgs, get_response

from .llm_judge import llm_judge
from .reward import (
    REWARD_REGISTRY,
    clear_reward_registry,
    compute_total_reward,
    is_precondition,
    is_sandbox_reward,
    reward,
)
from .tool import TOOL_REGISTRY, clear_tool_registry, is_sandbox_tool, tool

__all__ = [
    # User-facing API
    "reward",
    "tool",
    "RewardArgs",
    "ToolArgs",
    "get_response",
    "llm_judge",
    # Registries (used by CLI and trainer)
    "REWARD_REGISTRY",
    "TOOL_REGISTRY",
    "clear_reward_registry",
    "clear_tool_registry",
    "is_precondition",
    "is_sandbox_reward",
    "is_sandbox_tool",
    "compute_total_reward",
]
