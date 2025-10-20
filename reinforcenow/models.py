# reinforcenow/models.py

from enum import Enum
from pydantic import BaseModel, model_validator
from typing import Optional, List, TypeAlias, Literal

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


# ===== API Models =====

class DeviceCode(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int = 1800
    interval: int = 5


from typing import Optional

class Token(BaseModel):
    access_token: str
    organization_id: Optional[str] = None


class TokenError(BaseModel):
    error: str


class Organization(BaseModel):
    id: str
    name: str
    role: OrgRole


class Organizations(BaseModel):
    organizations: List[Organization]
    active_organization_id: Optional[str] = None


class TrainingParams(BaseModel):
    model: Literal[
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-4B-Instruct-2507",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-1B",
    ]
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 0.0001
    max_steps: Optional[int] = None
    qlora_rank: int = 32
    qlora_alpha: Optional[int] = None
    val_steps: Optional[int] = None
    val_epochs: Optional[int] = None
    save_steps: Optional[int] = None
    save_epochs: Optional[int] = None
    loss_fn: Optional[str] = None
    adv_estimator: Optional[str] = None
    kl_penalty_coef: float = 0.01
    strategy: Optional[Literal["sft", "rl"]] = None


class ProjectConfig(BaseModel):
    project_id: str
    project_name: str
    dataset_id: str
    dataset_type: DatasetType = DatasetType.RL
    organization_id: Optional[str] = None
    params: Optional[TrainingParams] = None

    @model_validator(mode='after')
    def validate_dataset_type(self):
        """Set mode based on dataset_type and validate RL parameters."""
        if self.params:
            # Set strategy based on dataset_type
            if self.dataset_type == DatasetType.SFT:
                self.params.strategy = "sft"
                # Clear RL-specific params for SFT
                self.params.loss_fn = None
                self.params.adv_estimator = None
                self.params.kl_penalty_coef = 0.0
            else:  # RL
                self.params.strategy = "rl"
                # Set RL defaults if not specified
                if self.params.loss_fn is None:
                    self.params.loss_fn = "importance_sampling"
                if self.params.adv_estimator is None:
                    self.params.adv_estimator = "grpo"
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