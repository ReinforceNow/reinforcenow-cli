# reinforcenow/models.py

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, model_validator


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
    model: ModelType = ModelType.QWEN3_8B
    qlora_rank: int = 32
    batch_size: int = 32
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs

    # Validation frequency (mutually exclusive)
    val_steps: Optional[int] = None  # Validate every N steps
    val_epochs: Optional[int] = None  # Validate every N epochs

    # Save frequency (mutually exclusive)
    save_steps: Optional[int] = None  # Save checkpoint every N steps
    save_epochs: Optional[int] = None  # Save checkpoint every N epochs

    # RL-specific parameters (only for dataset_type="rl")
    loss_fn: Optional[LossFunction] = None  # Loss function: ppo or importance_sampling
    adv_estimator: Optional[AdvantageEstimator] = None  # Advantage estimator: grpo, gae, or reinforce

    # KL penalty (only for RL)
    compute_post_kl: bool = False  # Compute KL divergence after training (default: False)
    kl_penalty_coef: float = 0.01  # KL penalty coefficient (default: 0.01)

    @model_validator(mode='after')
    def validate_exclusive(self):
        # Validate validation params
        if self.val_steps is not None and self.val_epochs is not None:
            raise ValueError("Cannot specify both val_steps and val_epochs - use one or the other")
        if self.val_steps is None and self.val_epochs is None:
            # Default to validating every 100 steps
            self.val_steps = 100

        # Validate save params
        if self.save_steps is not None and self.save_epochs is not None:
            raise ValueError("Cannot specify both save_steps and save_epochs - use one or the other")
        if self.save_steps is None and self.save_epochs is None:
            # Default to saving every epoch
            self.save_epochs = 1

        return self


class ProjectConfig(BaseModel):
    project_id: str
    project_name: str
    dataset_id: str
    dataset_type: DatasetType = DatasetType.RL  # SFT or RL
    organization_id: Optional[str] = None
    params: Optional[TrainingParams] = None

    @model_validator(mode='after')
    def validate_dataset_type(self):
        """Validate that RL parameters are only set for RL datasets."""
        if self.dataset_type == DatasetType.SFT and self.params:
            # For SFT, these RL-specific params should not be set
            rl_params = ['loss_fn', 'adv_estimator', 'compute_post_kl', 'kl_penalty_coef']
            for param in rl_params:
                if getattr(self.params, param, None) is not None:
                    # Reset RL params to None for SFT
                    setattr(self.params, param, None)
        return self


