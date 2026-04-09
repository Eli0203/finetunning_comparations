from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, field_validator, model_validator
from peft import TaskType  
import torch
from typing import Any, Optional

class Settings(BaseSettings):
    # 1. Security: Hugging Face Token (Protected as SecretStr)
    hf_token: str = Field(..., validation_alias="HF_TOKEN")
    
    # Model and Task Parameters
    model_id: str = Field(default="bert-base-uncased", validation_alias="MODEL_ID")
    task_name: str = Field(default="mrpc", validation_alias="TASK_NAME")
    task_type: TaskType = Field(default=TaskType.SEQ_CLS, validation_alias="TASK_TYPE")
    max_seq_length: int = Field(default=512, validation_alias="MAX_SEQ_LENGTH")
    output_dir: str = Field(default="./output", validation_alias="OUTPUT_DIR")
    
    # Training Hyperparameters
    learning_rate: float = Field(default=2e-5, validation_alias="LEARNING_RATE")
    batch_size: int = Field(default=16, validation_alias="BATCH_SIZE")
    epochs: int = Field(default=3, validation_alias="EPOCHS")   
    
    # LoRA configuration
    execute_lora: bool = Field(default=True, validation_alias="EXECUTE_LORA")
    lora_rank: int = Field(default=8, validation_alias="LORA_RANK")
    lora_alpha: int = Field(default=16, validation_alias="LORA_ALPHA")
    lora_dropout: float = Field(default=0.1, validation_alias="LORA_DROPOUT")
    execute_laplace: bool = Field(default=False, validation_alias="EXECUTE_LAPLACE")

    # Causal engine configuration (optional feature)
    execute_causal_engine: bool = Field(default=False, validation_alias="EXECUTE_CAUSAL_ENGINE")
    causal_sampler_mode: str = Field(default="gradient", validation_alias="CAUSAL_SAMPLER_MODE")
    random_dirichlet_init: bool = Field(default=False, validation_alias="RANDOM_DIRICHLET_INIT")
    enable_pg_pos: bool = Field(default=False, validation_alias="ENABLE_PG_POS")
    kfac_correlation: bool = Field(default=False, validation_alias="KFAC_CORRELATION")
    apply_interval: int = Field(default=10, gt=0, validation_alias="APPLY_INTERVAL")
    causal_softmax_temp_initial: float = Field(
        default=2.0,
        gt=0,
        validation_alias="CAUSAL_SOFTMAX_TEMP_INITIAL",
    )
    causal_softmax_temp_final: float = Field(
        default=0.5,
        gt=0,
        validation_alias="CAUSAL_SOFTMAX_TEMP_FINAL",
    )
    causal_temp_schedule: str = Field(
        default="linear_decay",
        validation_alias="CAUSAL_TEMP_SCHEDULE",
    )
    enable_interventional_weights: bool = Field(
        default=False,
        validation_alias="ENABLE_INTERVENTIONAL_WEIGHTS",
    )
    
    # Phase 6: Resource & Quantization Constraints
    max_ram_threshold_gb: float = Field(
        default=9.0,
        gt=0,
        lt=10,
        validation_alias="MAX_RAM_THRESHOLD_GB",
        description="Hard RAM ceiling in GB (must be < 10 GB per Principle IV)."
    )
    default_quantization: str = Field(
        default="auto",
        validation_alias="DEFAULT_QUANTIZATION",
        description="Default quantization strategy: 'auto', 'fp16', or 'nf4_forced'."
    )
    
    # QLoRA configuration
    execute_qlora: bool = Field(default=False, validation_alias="EXECUTE_QLORA")    
    qlora_rank: int = Field(default=4, validation_alias="QLORA_RANK")
    qlora_alpha: int = Field(default=8, validation_alias="QLORA_ALPHA")
    qlora_dropout: float = Field(default=0.1, validation_alias="QLORA_DROPOUT")
    
    # Training and Storage Strategies
    eval_strategy: str = Field(default="epoch", validation_alias="EVAL_STRATEGY")
    save_strategy: str = Field(default="epoch", validation_alias="SAVE_STRATEGY")
    save_total_limit: int = Field(default=2, validation_alias="SAVE_TOTAL_LIMIT")
    logging_strategy: str = Field(default="epoch", validation_alias="LOGGING_STRATEGY")
    logging_dir: str = Field(default="./logs", validation_alias="LOGGING_DIR")
    
    # Execution Logic
    use_mock_data: bool = Field(default=False, validation_alias="USE_MOCK_DATA")

    # 2. Portability: Automatic Device Detection
    @field_validator('causal_sampler_mode')
    @classmethod
    def validate_causal_sampler_mode(cls, v: str) -> str:
        """Validate causal sampler mode selection."""
        valid_modes = {'gradient', 'mixture_of_gaussians'}
        if v not in valid_modes:
            raise ValueError(
                f"causal_sampler_mode must be one of {valid_modes}, got '{v}'"
            )
        return v

    @field_validator('causal_temp_schedule')
    @classmethod
    def validate_causal_temp_schedule(cls, v: str) -> str:
        """Validate supported temperature schedule."""
        valid_schedules = {'linear_decay'}
        if v not in valid_schedules:
            raise ValueError(
                f"causal_temp_schedule must be one of {valid_schedules}, got '{v}'"
            )
        return v

    @field_validator('default_quantization')
    @classmethod
    def validate_default_quantization(cls, v: str) -> str:
        """Validate default quantization strategy."""
        valid_strategies = {'auto', 'fp16', 'nf4_forced'}
        if v not in valid_strategies:
            raise ValueError(
                f"default_quantization must be one of {valid_strategies}, got '{v}'"
            )
        return v

    @property
    def device(self) -> str:
        """Determina el hardware disponible para ejecución optimizada [6, 7]."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @model_validator(mode='after')
    def validate_temperature_constraints(self) -> 'Settings':
        """Validate causal temperature field constraints per Phase 6 spec."""
        # Constraint 1: causal_softmax_temp_initial >= causal_softmax_temp_final
        if self.causal_softmax_temp_initial < self.causal_softmax_temp_final:
            raise ValueError(
                f"causal_softmax_temp_initial ({self.causal_softmax_temp_initial}) must be >= "
                f"causal_softmax_temp_final ({self.causal_softmax_temp_final})"
            )
        # Constraint 2: causal_softmax_temp_final > 0.1
        if self.causal_softmax_temp_final <= 0.1:
            raise ValueError(
                f"causal_softmax_temp_final ({self.causal_softmax_temp_final}) must be > 0.1"
            )
        return self

    # .env file configuration
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore',
        populate_by_name=True,
    )


class CausalTrainingConfig(BaseModel):
    """
    Configuration for causal training orchestration.
    
    This config defines all parameters needed for CausalTrainingOrchestrator
    to coordinate causal weight sampling, application, and monitoring during
    training.
    
    Core Parameters:
    - total_causal_budget: Total number of causal samples during training
    - async_max_steps: Maximum async sampling iterations
    - apply_interval: Apply weights every N training steps (rate limiting)
    - device: Computation device (auto-detected or override)
    
    Additional Parameters (for Phase 5 compatibility):
    - seed: Random seed for reproducibility
    - warmup_steps: Number of warmup steps before main training
    - enable_warmup: Whether to run warmup phase
    - logging_level: Logging verbosity level
    """
    
    # Core causal training parameters
    total_causal_budget: int = Field(
        default=1000, 
        description="Total number of causal samples for training budget",
        gt=0  # Greater than 0 validation
    )
    async_max_steps: int = Field(
        default=100, 
        description="Maximum number of async sampling iterations",
        gt=0  # Greater than 0 validation
    )
    apply_interval: int = Field(
        default=10, 
        description="Apply weights every N training steps (rate limiting)",
        gt=0  # Greater than 0 validation
    )
    device: str = Field(
        default="cpu", 
        description="Computation device ('cpu', 'cuda', 'mps', or auto)"
    )
    
    # Additional parameters for reproducibility and Phase 5 features
    seed: int = Field(
        default=42, 
        description="Random seed for reproducibility in causal sampling",
        ge=0  # Greater than or equal to 0
    )
    warmup_steps: Optional[int] = Field(
        default=None, 
        description="Number of warmup steps before main training (None = no warmup)",
        ge=1  # If provided, must be >= 1
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
    # Proposed phase-aware parameters (kept in parallel with legacy fields)
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
        """Ensure temperature_min < temperature when annealing is enabled."""
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
        """
        Post-initialization validation and setup.
        
        Ensures device detection works if 'auto' is specified.
        """
        if self.device == 'auto':
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


settings = Settings()