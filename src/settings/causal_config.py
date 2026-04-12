"""CausalTrainingConfig for orchestrating causal training phases and budget allocation.

Single responsibility: Configuration for causal-specific parameters including
budget, temperature annealing, warmup phases, and PAC bounds.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import torch


class CausalTrainingConfig(BaseModel):
    """Configuration for causal training orchestration.
    
    Defines all parameters for CausalTrainingOrchestrator to coordinate
    causal weight sampling, application, and monitoring during training.
    """
    
    # Core causal training parameters
    total_causal_budget: int = Field(
        default=1000, 
        description="Total number of causal samples for training budget",
        gt=0
    )
    async_max_steps: int = Field(
        default=100, 
        description="Maximum number of async sampling iterations",
        gt=0
    )
    apply_interval: int = Field(
        default=10, 
        description="Apply weights every N training steps (rate limiting)",
        gt=0
    )
    device: str = Field(
        default="cpu", 
        description="Computation device ('cpu', 'cuda', 'mps', or auto)"
    )
    
    # Additional parameters for reproducibility and Phase 5 features
    seed: int = Field(
        default=42, 
        description="Random seed for reproducibility in causal sampling",
        ge=0
    )
    warmup_steps: Optional[int] = Field(
        default=None, 
        description="Number of warmup steps before main training (None = no warmup)",
        ge=1
    )
    enable_warmup: bool = Field(
        default=False, 
        description="Whether to enable warmup phase before main training"
    )
    enable_interventional_weights: bool = Field(
        default=False,
        description="Enable interventional sample reweighting callback during training.",
    )
    logging_level: str = Field(
        default="INFO", 
        description="Logging verbosity level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Temperature-scaled softmax budget allocation
    causal_softmax_temp_init: float = Field(
        default=2.0,
        description="Initial τ during causal initiation phase (<20% progress).",
        gt=0,
    )
    causal_softmax_temp_final: float = Field(
        default=0.5,
        description="Final τ during convergence phase (>80% progress).",
        gt=0,
    )
    causal_temp_annealing: bool = Field(
        default=True,
        description="Enable phase-based τ annealing across training.",
    )
    causal_temp_decay_strategy: str = Field(
        default="linear",
        description="Decay curve for transition phase: linear or cubic.",
    )

    # Legacy temperature fields (backward compatibility)
    temperature: float = Field(
        default=1.0,
        description=(
            "Softmax temperature τ for budget allocation (must be > 0). "
            "1.0 uses EqualBudgetAllocationStrategy; any other value activates "
            "TemperatureSoftmaxAllocationStrategy. Higher τ → more uniform weights; "
            "lower τ → sharper concentration on high-score paths."
        ),
        gt=0,
    )
    temperature_anneal: bool = Field(
        default=False,
        description=(
            "Enable linear temperature annealing. When True, τ decays linearly "
            "from `temperature` to `temperature_min` over the training run."
        ),
    )
    temperature_min: float = Field(
        default=0.1,
        description=(
            "Minimum temperature for the annealing schedule (must be > 0). "
            "Ignored when temperature_anneal=False."
        ),
        gt=0,
    )

    # Warm-up plateau detection
    warmup_plateau_delta: float = Field(
        default=1e-4,
        description=(
            "Minimum absolute loss decrease required to avoid triggering the "
            "plateau counter during warm-up. Set to 0 to disable plateau detection."
        ),
        ge=0,
    )
    warmup_plateau_patience: int = Field(
        default=3,
        description=(
            "Number of consecutive warm-up steps that must show no improvement "
            "greater than warmup_plateau_delta before early exit is triggered."
        ),
        ge=1,
    )
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that device is a supported option."""
        valid_devices = {'cpu', 'cuda', 'mps', 'auto'}
        if v not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got '{v}'"
            )
        return v
    
    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate that logging level is supported."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(
                f"logging_level must be one of {valid_levels}, got '{v}'"
            )
        return v.upper()
    
    @field_validator('warmup_steps')
    @classmethod
    def validate_warmup_steps(cls, v: Optional[int]) -> Optional[int]:
        """Validate warmup_steps if provided."""
        if v is not None and v < 1:
            raise ValueError(f"warmup_steps must be >= 1 if provided, got {v}")
        return v

    @field_validator('causal_temp_decay_strategy')
    @classmethod
    def validate_causal_temp_decay_strategy(cls, v: str) -> str:
        """Validate phase transition decay strategy for causal temperature."""
        valid_strategies = {'linear', 'cubic'}
        if v not in valid_strategies:
            raise ValueError(
                f"causal_temp_decay_strategy must be one of {valid_strategies}, got '{v}'"
            )
        return v

    @model_validator(mode='after')
    def validate_temperature_annealing(self) -> 'CausalTrainingConfig':
        """Ensure temperature constraints are satisfied."""
        if self.temperature_anneal and self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be strictly less than "
                f"temperature ({self.temperature}) when temperature_anneal=True."
            )
        if self.causal_temp_annealing and self.causal_softmax_temp_final >= self.causal_softmax_temp_init:
            raise ValueError(
                "causal_softmax_temp_final must be strictly less than "
                "causal_softmax_temp_init when causal_temp_annealing=True."
            )
        return self

    def get_causal_temperature(self, progress_ratio: float) -> float:
        """Compute phase-aware causal softmax temperature τ.

        Phase policy:
        - MAP warm-up: not applicable (handled outside this scheduler)
        - Causal initiation (<20%): fixed τ_init for exploration
        - Transition (20%-80%): linear/cubic decay
        - Convergence (>80%): fixed τ_final for exploitation
        """
        if not self.causal_temp_annealing:
            return self.causal_softmax_temp_init

        p = min(1.0, max(0.0, progress_ratio))
        if p < 0.2:
            return self.causal_softmax_temp_init
        if p > 0.8:
            return self.causal_softmax_temp_final

        transition_progress = (p - 0.2) / 0.6
        if self.causal_temp_decay_strategy == 'cubic':
            transition_progress = transition_progress ** 3

        return self.causal_softmax_temp_init + (
            self.causal_softmax_temp_final - self.causal_softmax_temp_init
        ) * transition_progress
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation and device auto-detection."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
