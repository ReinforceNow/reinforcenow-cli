# reinforcenow/models.py

from enum import Enum
from pydantic import BaseModel, model_validator
from typing import Optional, List, Literal


# ===== Enums =====

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
    # Core parameters
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
    ] = "meta-llama/Llama-3.2-1B"  # Default to smallest model
    mode: Optional[str] = None  # "sft" or "rl" - will be set based on dataset_type

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    max_steps: Optional[int] = None  # If set, overrides num_epochs

    # LoRA parameters
    qlora_rank: int = 32
    qlora_alpha: Optional[int] = None  # Default to 2 * qlora_rank

    # Validation and checkpointing (mutually exclusive pairs)
    val_steps: Optional[int] = None  # Validate every N steps
    val_epochs: Optional[int] = None  # Validate every N epochs
    save_steps: Optional[int] = None  # Save checkpoint every N steps
    save_epochs: Optional[int] = None  # Save checkpoint every N epochs

    # RL-specific parameters
    loss_fn: Optional[str] = None  # "ppo" or "importance_sampling"
    adv_estimator: Optional[str] = None  # "grpo", "gae", or "reinforce"
    kl_penalty_coef: float = 0.01  # KL penalty coefficient
    training_mode: str = "sync"  # "sync", "async", or "stream_minibatch"

    # Optional parameters
    gradient_checkpointing: bool = True
    fp16: bool = True
    num_workers: int = 4
    checkpoint_dir: str = "/workspace/checkpoints"

    @model_validator(mode='after')
    def validate_and_set_defaults(self):
        # Set qlora_alpha default
        if self.qlora_alpha is None:
            self.qlora_alpha = self.qlora_rank * 2

        # Validate validation params
        if self.val_steps is not None and self.val_epochs is not None:
            raise ValueError("Cannot specify both val_steps and val_epochs - use one or the other")

        # Validate save params
        if self.save_steps is not None and self.save_epochs is not None:
            raise ValueError("Cannot specify both save_steps and save_epochs - use one or the other")

        # Set save_epochs default if nothing specified
        if self.save_steps is None and self.save_epochs is None:
            self.save_epochs = 1

        # Validate save_epochs doesn't exceed num_epochs
        if self.save_epochs and self.save_epochs > self.num_epochs:
            raise ValueError(f"save_epochs ({self.save_epochs}) cannot exceed num_epochs ({self.num_epochs})")

        return self


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
            # Set mode based on dataset_type
            if self.dataset_type == DatasetType.SFT:
                self.params.mode = "sft"
                # Clear RL-specific params for SFT
                self.params.loss_fn = None
                self.params.adv_estimator = None
                self.params.kl_penalty_coef = 0.0
            else:  # RL
                self.params.mode = "rl"
                # Set RL defaults if not specified
                if self.params.loss_fn is None:
                    self.params.loss_fn = "importance_sampling"
                if self.params.adv_estimator is None:
                    self.params.adv_estimator = "grpo"
        return self


