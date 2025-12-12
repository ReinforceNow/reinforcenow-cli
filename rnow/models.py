"""
ReinforceNow Models - User-facing types for the rnow CLI.

This module contains ONLY the types that users need:
- RewardArgs for reward function signatures
- ProjectConfig and related configs for config.yml
- Basic enums

Trainer-internal types (Env, StepResult, Observation) live in docker/trainer/
where tinker is available.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelType(str, Enum):
    QWEN3_8B = "qwen3-8b"
    GLM4_9B = "glm4-9b"


class OrgRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class DatasetType(str, Enum):
    SFT = "sft"  # Supervised Fine-Tuning
    RL = "rl"  # Reinforcement Learning


class LossFunction(str, Enum):
    PPO = "ppo"  # Proximal Policy Optimization
    IS = "importance_sampling"  # Importance Sampling


class AdvantageEstimator(str, Enum):
    GRPO = "grpo"  # Generalized Reward Policy Optimization
    GAE = "gae"  # Generalized Advantage Estimation
    REINFORCE = "reinforce"  # REINFORCE algorithm


class TerminationPolicy(str, Enum):
    MAX_TURNS = "max_turns"  # Episode ends when max_turns is exhausted
    LAST_TOOL = "last_tool"  # Episode ends when assistant responds without a tool call


class RewardArgs(BaseModel):
    """Arguments passed to reward functions containing context about the sample."""

    metadata: dict = Field(default_factory=dict)
    variables: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class DeviceCode(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int = 1800
    interval: int = 5


class Token(BaseModel):
    access_token: str
    organization_id: str | None = None


class TokenError(BaseModel):
    error: str


class Organization(BaseModel):
    id: str
    name: str
    role: OrgRole


class Organizations(BaseModel):
    organizations: list[Organization]
    active_organization_id: str | None = None


# Supported model IDs
SUPPORTED_MODELS = Literal[
    # Qwen models
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    # OpenAI models
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # DeepSeek models
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.1-Base",
    # Meta Llama models
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
]

# Maximum context window for all supported models
MAX_CONTEXT_WINDOW = 32768

# Conservative max_tokens limit (leaves room for prompts)
MAX_GENERATION_TOKENS = 30000

# Maximum LoRA rank per model
# Models not listed here default to 128
MODEL_MAX_LORA_RANK: dict[str, int] = {
    # Max 32
    "openai/gpt-oss-120b": 32,
    "openai/gpt-oss-20b": 32,
    # Max 64
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 64,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 64,
    "Qwen/Qwen3-30B-A3B": 64,
    "Qwen/Qwen3-30B-A3B-Base": 64,
    "deepseek-ai/DeepSeek-V3.1": 64,
    "deepseek-ai/DeepSeek-V3.1-Base": 64,
    # Max 128 (default for all others)
    # Qwen/Qwen3-32B, Qwen/Qwen3-8B*, Qwen/Qwen3-4B-Instruct-2507, all meta-llama/*
}

# Default max LoRA rank for models not in the dict
DEFAULT_MAX_LORA_RANK = 128


def get_max_lora_rank(model_path: str) -> int:
    """Get the maximum LoRA rank for a given model."""
    return MODEL_MAX_LORA_RANK.get(model_path, DEFAULT_MAX_LORA_RANK)


class DataConfig(BaseModel):
    """Data configuration for training."""

    model_config = ConfigDict(extra="forbid")

    train_file: str = "train.jsonl"
    batch_size: int = Field(..., gt=0, le=32)  # Max 32
    group_size: int = Field(default=4, gt=0, le=64)  # Max 64, RL only
    val_split: float | None = Field(default=None, ge=0, le=1)  # Validation split ratio (0.0-1.0)

    @model_validator(mode="after")
    def _check_batch_group_product(self):
        """Validate batch_size * group_size <= 2048 (sandbox concurrency limit)."""
        prod = self.batch_size * self.group_size
        if prod > 2048:
            raise ValueError(
                f"batch_size * group_size must be <= 2048 (got {self.batch_size} * {self.group_size} = {prod})"
            )
        return self


class ModelConfig(BaseModel):
    """Model configuration.

    The `path` field accepts either:
    - A supported base model name (e.g., "Qwen/Qwen3-4B-Instruct-2507")
    - A ReinforceNow model ID (e.g., "acfa2862-23a9-4e65-ab68-b9b2698b0e75") to resume from a finetuned model
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: str = Field(
        ...,
        description="Base model name (e.g., 'Qwen/Qwen3-8B') or a ReinforceNow model ID to resume from",
    )
    qlora_rank: int = Field(default=32, ge=1)
    qlora_alpha: int | None = Field(default=None, ge=1)  # Defaults to qlora_rank * 2
    name: str | None = None  # Custom name for the output model (default: auto-generated)
    description: str | None = None  # Custom description for the output model

    # Internal fields resolved by the server (not set by users)
    resolved_checkpoint_path: str | None = Field(
        default=None,
        alias="_resolvedCheckpointPath",
        description="Internal: Tinker checkpoint path resolved from model ID",
    )
    resolved_base_model: str | None = Field(
        default=None,
        alias="_baseModelName",
        description="Internal: Base model name resolved from model ID",
    )


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for RL training."""

    model_config = ConfigDict(extra="forbid")

    loss_fn: Literal["ppo", "importance_sampling"] = "ppo"
    adv_estimator: Literal["grpo", "gae", "reinforce"] = "grpo"
    kl_penalty_coef: float = Field(default=0.01, ge=0)


class RolloutConfig(BaseModel):
    """Rollout configuration for RL training."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=1, gt=0)
    max_tokens: int = Field(default=2048, gt=0)
    termination_policy: Literal["max_turns", "last_tool"] = "last_tool"
    thinking_mode: Literal["disabled", "easy", "medium", "hard"] | None = (
        None  # None = model default
    )
    mcp_url: str | list[str] | None = Field(
        default=None,
        description="MCP server URL(s) for tools. Can be a single URL or a list of URLs. Can be used alongside env.py to combine both tool sources.",
    )
    max_tool_response_chars: int | None = Field(
        default=4000,
        gt=0,
        description="Maximum characters for tool responses. Longer responses are truncated. Set to null/None to disable truncation.",
    )


class TrainerConfig(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(..., gt=0)
    learning_rate: float = Field(default=0.0001, gt=0)
    save_step: int = Field(default=0, ge=0)  # Save checkpoint every N steps (0 = end of epoch only)
    eval_step: int = Field(default=0, ge=0)  # Evaluate every N steps (0 = end of epoch only)


class ProjectConfig(BaseModel):
    """Full project configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(default="")
    project_name: str = Field(default="")
    dataset_id: str = Field(default="")
    dataset_name: str | None = None
    dataset_type: DatasetType = Field(...)
    organization_id: str | None = None

    # Nested config sections
    data: DataConfig = Field(...)
    model: ModelConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    algorithm: AlgorithmConfig | None = None  # RL only
    rollout: RolloutConfig | None = None  # RL only

    @model_validator(mode="after")
    def validate_config(self):
        """Set defaults and validate based on dataset_type."""
        if self.dataset_type == DatasetType.RL:
            # Set RL defaults if not specified
            if self.algorithm is None:
                self.algorithm = AlgorithmConfig()
            if self.rollout is None:
                self.rollout = RolloutConfig()
        else:  # SFT
            # Clear RL-specific configs for SFT
            self.algorithm = None
            self.rollout = None

        # Validate qlora_rank against model-specific limits
        # Only validate for known models (model IDs for finetuned models are validated server-side)
        model_path = self.model.path
        if "/" in model_path:  # Standard model path (not a model ID)
            max_rank = get_max_lora_rank(model_path)
            if self.model.qlora_rank > max_rank:
                raise ValueError(
                    f"qlora_rank {self.model.qlora_rank} exceeds maximum {max_rank} for model {model_path}"
                )

        return self
