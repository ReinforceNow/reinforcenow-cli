"""
ReinforceNow Models - User-facing types for the rnow CLI.

This module contains ONLY the types that users need:
- RewardArgs for reward function signatures
- ProjectConfig and related configs for config.yml
- Basic enums

Trainer-internal types (Env, StepResult, Observation) live in docker/trainer/.
"""

from __future__ import annotations

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
    SFT = "sft"  # Supervised Finetuning
    RL = "rl"  # Reinforcement Learning
    DISTILL = "distill"  # On-policy Distillation
    MIDTRAIN = "midtrain"  # Continued Pretraining on raw text


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
    secrets: dict = Field(
        default_factory=dict
    )  # User-defined secrets from .env file or project settings

    class Config:
        arbitrary_types_allowed = True


def get_response(messages: list) -> str:
    """Extract text content from the last assistant message.

    Handles both string and list content (ContentPart format for multi-part responses).
    Only returns "text" type parts - thinking/reasoning is excluded to avoid
    intermediate \\boxed{} expressions confusing parsers like math-verify.

    Note: This finds the LAST assistant message, skipping any trailing tool messages.
    This is important for agentic workflows where tool results follow assistant messages.

    Example:
        @reward
        def my_reward(args: RewardArgs, messages: list) -> float:
            response = get_response(messages)
            return 1.0 if "answer" in response else 0.0
    """
    # Find the last assistant message (skip trailing tool messages)
    content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            break

    if isinstance(content, list):
        # ContentPart format: [{type: "thinking", thinking: "..."}, {type: "text", text: "..."}]
        # Only extract "text" type parts - skip "thinking" parts which contain intermediate reasoning
        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
        if text_parts:
            return "".join(text_parts)
        # Fallback for backwards compatibility: if no text parts, return all content
        return "".join(p.get("text", "") or p.get("thinking", "") for p in content)
    return content or ""


# --- train.jsonl validation models ---


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields like tool_calls

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list  # str for text, list for multimodal (VLM images)


class TrainEntry(BaseModel):
    """A single entry in train.jsonl."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields like variables, metadata

    messages: list[Message] = Field(..., min_length=1)
    rewards: list[str] | None = None  # Required for RL, optional for SFT
    tools: list[str] | None = None  # Optional: filter which tools are available
    docker: str | None = None  # Optional: Docker image for isolated sandbox
    docker_env: dict[str, str] | None = None  # Optional: Environment variables for sandbox
    metadata: dict | None = None
    variables: dict | None = None

    @model_validator(mode="after")
    def validate_messages_not_empty(self):
        if not self.messages:
            raise ValueError("messages list cannot be empty")
        return self

    @model_validator(mode="after")
    def validate_docker_fields(self):
        """Validate that docker_env requires docker to be set."""
        if self.docker_env and not self.docker:
            raise ValueError("docker_env requires docker field to be set")
        return self


class TrainEntryRL(TrainEntry):
    """Train entry for RL datasets - rewards field is required."""

    rewards: list[str] = Field(..., min_length=1)


class TrainEntryMidtrain(BaseModel):
    """Train entry for midtrain - raw text for continued pretraining."""

    model_config = ConfigDict(extra="allow")
    text: str = Field(..., min_length=1)


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


# Supported model IDs (as set for runtime validation)
SUPPORTED_MODELS_SET: set[str] = {
    # Qwen models (text)
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    # Qwen models (vision)
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # OpenAI models (reasoning)
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
    # Moonshot models (reasoning)
    "moonshotai/Kimi-K2-Thinking",
}

# Type alias for supported models (for type hints)
SUPPORTED_MODELS = Literal[
    # Qwen models (text)
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    # Qwen models (vision)
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # OpenAI models (reasoning)
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
    # Moonshot models (reasoning)
    "moonshotai/Kimi-K2-Thinking",
]

# Maximum context window for all supported models
MAX_CONTEXT_WINDOW = 32768

# Conservative max_tokens limit (leaves room for prompts)
MAX_GENERATION_TOKENS = 30000

# Models that do NOT support tool calling
# - gpt-oss models use GptOssRenderer which doesn't support tools
# - Base/non-instruct models use RoleColonRenderer which doesn't support tools
MODELS_WITHOUT_TOOL_SUPPORT: set[str] = {
    # OpenAI reasoning models (GptOssRenderer)
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # Base models (RoleColonRenderer)
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-8B-Base",
    "deepseek-ai/DeepSeek-V3.1-Base",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
}


def supports_tool_calling(model_path: str) -> bool:
    """Check if a model supports tool calling."""
    return model_path not in MODELS_WITHOUT_TOOL_SUPPORT


# Maximum LoRA rank per model
# Models not listed here default to 128
MODEL_MAX_LORA_RANK: dict[str, int] = {
    # Max 32 (reasoning models)
    "openai/gpt-oss-120b": 32,
    "openai/gpt-oss-20b": 32,
    "moonshotai/Kimi-K2-Thinking": 32,
    # Max 64 (large MoE models)
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 64,
    "Qwen/Qwen3-VL-235B-A22B-Instruct": 64,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 64,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 64,
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
        description="Internal: Checkpoint path resolved from model ID",
    )
    resolved_base_model: str | None = Field(
        default=None,
        alias="_baseModelName",
        description="Internal: Base model name resolved from model ID",
    )


class TeacherConfig(BaseModel):
    """Teacher model configuration for on-policy distillation.

    Only available when dataset_type: distill.
    Teacher must be a supported HuggingFace model ID (e.g., 'Qwen/Qwen3-32B').
    API models like GPT-4o or Claude are NOT supported for on-policy distillation.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Teacher model path (e.g., 'Qwen/Qwen3-32B')")
    agentic: bool = Field(
        default=False,
        description="Enable agentic distillation with multi-turn tool support. Requires tools.py and rollout config.",
    )

    @model_validator(mode="after")
    def validate_teacher_path(self):
        """Validate that teacher path is a supported model."""
        if self.path not in SUPPORTED_MODELS_SET:
            raise ValueError(
                f"Unsupported teacher model: {self.path}. "
                f"On-policy distillation requires a supported HuggingFace model ID. "
                f"Supported models: {', '.join(sorted(SUPPORTED_MODELS_SET))}"
            )
        return self


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
    max_context_window: int = Field(
        default=32768,
        gt=0,
        description="Maximum context window in tokens. Rollouts are marked as truncated if conversation exceeds this. Default 32768 (32k).",
    )
    termination_policy: Literal["max_turns", "last_tool"] = "last_tool"
    reasoning_mode: Literal["disabled", "low", "medium", "high"] | None = Field(
        default=None,
        description="Reasoning mode for models that support it. Controls chain-of-thought reasoning. "
        "Values: 'disabled' (no reasoning), 'low', 'medium', 'high'. Default None (model default). "
        "For GPU models (gpt-oss, Qwen3, DeepSeek): controls thinking/reasoning effort. "
        "For OpenAI API (GPT-5): maps to reasoning effort parameter.",
    )
    mcp_url: str | list[str] | None = Field(
        default=None,
        description="MCP server URL(s) for tools. Can be a single URL or a list of URLs. Can be used alongside tools.py to combine both tool sources.",
    )
    tool_timeout: int = Field(
        default=60,
        gt=0,
        description="Timeout in seconds for tool calls. Browser automation may need longer timeouts (default: 60s).",
    )
    max_tool_response: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens for tool responses. Responses longer than this are truncated (expected behavior, not marked as truncated). Default None (no limit).",
    )
    include_thinking: bool = Field(
        default=False,
        description="Whether to include <think>...</think> blocks in messages passed to reward functions. Default is False (thinking is stripped).",
    )
    rollout_timeout: int | None = Field(
        default=None,
        gt=0,
        description="Maximum time in seconds for a single rollout. If exceeded, rollout is marked as timed out with 0 rewards. Default None (no timeout).",
    )


class TrainerConfig(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(..., gt=0)
    learning_rate: float = Field(default=0.0001, gt=0)
    save_step: int = Field(default=-1, ge=-1)  # -1 = end only, 0 = never save, N = every N steps
    eval_step: int = Field(default=0, ge=0)  # Evaluate every N steps (0 = end of epoch only)


class ProjectConfig(BaseModel):
    """Full project configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(default="")
    project_name: str = Field(default="")
    description: str | None = None  # Optional project description
    dataset_type: DatasetType = Field(...)
    organization_id: str | None = None

    # Nested config sections
    data: DataConfig = Field(...)
    model: ModelConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    algorithm: AlgorithmConfig | None = None  # RL only
    rollout: RolloutConfig | None = None  # RL and DISTILL
    teacher: TeacherConfig | None = None  # DISTILL only

    @model_validator(mode="after")
    def validate_config(self):
        """Set defaults and validate based on dataset_type."""
        if self.dataset_type == DatasetType.RL:
            # Set RL defaults if not specified
            if self.algorithm is None:
                self.algorithm = AlgorithmConfig()
            if self.rollout is None:
                self.rollout = RolloutConfig()
            # Teacher is only for distillation
            if self.teacher is not None:
                raise ValueError(
                    "'teacher' is only available for dataset_type: distill. "
                    "For RL training, remove the 'teacher' section from config.yml."
                )

        elif self.dataset_type == DatasetType.DISTILL:
            # Distillation requires teacher (path validation is in TeacherConfig)
            if self.teacher is None:
                raise ValueError("dataset_type: distill requires 'teacher' section")

            # Distillation needs rollout config for sampling
            if self.rollout is None:
                self.rollout = RolloutConfig()

            # Set algorithm defaults for distillation (KL penalty settings)
            if self.algorithm is None:
                self.algorithm = AlgorithmConfig()

        elif self.dataset_type == DatasetType.MIDTRAIN:
            # Midtrain is like SFT - no RL-specific configs
            if self.teacher is not None:
                raise ValueError(
                    "'teacher' is not available for dataset_type: midtrain. "
                    "For midtrain, remove the 'teacher' section from config.yml."
                )
            # Clear RL-specific configs for midtrain
            self.algorithm = None
            self.rollout = None

        else:  # SFT
            # Teacher is only for distillation
            if self.teacher is not None:
                raise ValueError(
                    "'teacher' is only available for dataset_type: distill. "
                    "For SFT training, remove the 'teacher' section from config.yml."
                )
            # Clear RL-specific configs for SFT
            self.algorithm = None
            self.rollout = None

        # Validate model path and qlora_rank
        # Only validate for standard model paths (model IDs for finetuned models are validated server-side)
        model_path = self.model.path
        if "/" in model_path:  # Standard model path (not a model ID)
            # Check if model is supported
            if model_path not in SUPPORTED_MODELS_SET:
                raise ValueError(
                    f"Unsupported model: {model_path}. "
                    f"Supported models: {', '.join(sorted(SUPPORTED_MODELS_SET))}"
                )

            # Validate qlora_rank against model-specific limits
            max_rank = get_max_lora_rank(model_path)
            if self.model.qlora_rank > max_rank:
                raise ValueError(
                    f"qlora_rank {self.model.qlora_rank} exceeds maximum {max_rank} for model {model_path}"
                )

        return self
