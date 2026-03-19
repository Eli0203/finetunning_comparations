from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, field_validator
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
    @property
    def device(self) -> str:
        """Determina el hardware disponible para ejecución optimizada [6, 7]."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # .env file configuration
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'
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
    logging_level: str = Field(
        default="INFO", 
        description="Logging verbosity level (DEBUG, INFO, WARNING, ERROR)"
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