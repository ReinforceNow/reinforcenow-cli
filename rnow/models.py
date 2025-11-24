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


class TrainingParams(BaseModel):
    model: Literal[
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
    batch_size: int = Field(...)
    num_epochs: int = Field(...)
    learning_rate: float = 0.0001
    max_tokens: int | None = 2048
    qlora_rank: int = 32
    kl_penalty_coef: float = 0.01
    max_turns: int | None = 1
    group_size: int | None = 4  # Number of parallel environments per prompt (RL only)
    qlora_alpha: int | None = None
    eval_step: int | None = None
    save_step: int | None = None
    val_split: float | None = None
    loss_fn: str | None = None
    adv_estimator: str | None = None


class ProjectConfig(BaseModel):
    project_id: str = Field(...)
    project_name: str = Field(...)
    dataset_id: str = Field(...)
    dataset_type: DatasetType = Field(...)
    organization_id: str | None = None
    params: TrainingParams = Field(...)

    @model_validator(mode="after")
    def validate_dataset_type(self):
        """Set mode based on dataset_type and validate RL parameters."""
        if self.params:
            # 1) Set strategy based on dataset_type
            if self.dataset_type == DatasetType.SFT:
                # Clear RL-specific params for SFT
                self.params.loss_fn = None
                self.params.adv_estimator = None
                self.params.kl_penalty_coef = 0.0
            else:  # RL
                # Set RL defaults if not specified
                if self.params.loss_fn is None:
                    self.params.loss_fn = "ppo"
                if self.params.adv_estimator is None:
                    self.params.adv_estimator = "grpo"

            # 2) Epoch eval and save
            if self.params.eval_step is None:
                self.params.eval_step = 0
            if self.params.save_step is None:
                self.params.save_step = 0

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
