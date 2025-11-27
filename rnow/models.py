from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import List, TypeAlias, Literal

from dataclasses import dataclass, field
import tinker
from abc import ABC, abstractmethod


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
    RL = "rl"    # Reinforcement Learning


class LossFunction(str, Enum):
    PPO = "ppo"  # Proximal Policy Optimization
    IS = "importance_sampling"  # Importance Sampling


class AdvantageEstimator(str, Enum):
    GRPO = "grpo"  # Generalized Reward Policy Optimization
    GAE = "gae"    # Generalized Advantage Estimation
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
    organizations: List[Organization]
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


class DataConfig(BaseModel):
    """Data configuration for training."""
    train_file: str = "train.jsonl"
    batch_size: int = Field(...)
    group_size: int = 4  # Number of parallel rollouts per prompt (RL only)
    val_split: float | None = None  # Validation split ratio (SFT only)


class ModelConfig(BaseModel):
    """Model configuration."""
    path: SUPPORTED_MODELS = Field(...)
    qlora_rank: int = 32
    qlora_alpha: int | None = None  # Defaults to qlora_rank * 2


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for RL training."""
    loss_fn: Literal["ppo", "importance_sampling"] = "ppo"
    adv_estimator: Literal["grpo", "gae", "reinforce"] = "grpo"
    kl_penalty_coef: float = 0.01


class RolloutConfig(BaseModel):
    """Rollout configuration for RL training."""
    max_turns: int = 1
    max_tokens: int = 2048
    termination_policy: Literal["max_turns", "last_tool"] = "last_tool"
    thinking_mode: Literal["none", "disabled", "easy", "medium", "hard"] = "none"


class TrainerConfig(BaseModel):
    """Trainer configuration."""
    num_epochs: int = Field(...)
    learning_rate: float = 0.0001
    save_step: int = 0  # Save checkpoint every N steps (0 = end of epoch only)
    eval_step: int = 0  # Evaluate every N steps (0 = end of epoch only)


class ProjectConfig(BaseModel):
    """Full project configuration."""
    project_id: str = Field(default="")
    project_name: str = Field(default="")
    dataset_id: str = Field(default="")
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
        """Set defaults based on dataset_type."""
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

        return self


StopCondition: TypeAlias = list[str] | list[int]

Action: TypeAlias = list[int]

Metrics: TypeAlias = dict[str, float | int]

Observation: TypeAlias = tinker.ModelInput


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


class Env(ABC):
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        pass
