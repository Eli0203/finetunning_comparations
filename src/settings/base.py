"""Shared base Settings model with common fields and validation logic."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from peft import TaskType  
import torch


class Settings(BaseSettings):
    """Base settings for all fine-tuning configurations.
    
    Shared across LoRA, Laplace-LoRA, and Causal-LoRA experiments.
    """
    
    # 1. Security: Hugging Face Token
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

    # Causal engine configuration
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
    
    # Resource & Quantization Constraints
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
    warmup_steps: int = Field(default=100, gt=0, validation_alias="WARMUP_STEPS")
    
    # Training and Storage Strategies
    eval_strategy: str = Field(default="epoch", validation_alias="EVAL_STRATEGY")
    save_strategy: str = Field(default="epoch", validation_alias="SAVE_STRATEGY")
    save_total_limit: int = Field(default=2, validation_alias="SAVE_TOTAL_LIMIT")
    logging_strategy: str = Field(default="epoch", validation_alias="LOGGING_STRATEGY")
    logging_dir: str = Field(default="./logs", validation_alias="LOGGING_DIR")
    
    # Execution Logic
    use_mock_data: bool = Field(default=False, validation_alias="USE_MOCK_DATA")

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

    @model_validator(mode='after')
    def validate_engine_selection(self) -> 'Settings':
        """Ensure mutually exclusive engine selection for causal and laplace modes."""
        if self.execute_causal_engine and self.execute_laplace:
            raise ValueError(
                "execute_causal_engine and execute_laplace cannot both be True"
            )
        return self

    @property
    def device(self) -> str:
        """Determine available hardware for optimized execution."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @model_validator(mode='after')
    def validate_temperature_constraints(self) -> 'Settings':
        """Validate causal temperature field constraints per Phase 6 spec."""
        if self.causal_softmax_temp_initial < self.causal_softmax_temp_final:
            raise ValueError(
                f"causal_softmax_temp_initial ({self.causal_softmax_temp_initial}) must be >= "
                f"causal_softmax_temp_final ({self.causal_softmax_temp_final})"
            )
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
