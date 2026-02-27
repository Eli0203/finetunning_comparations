from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from peft import TaskType  
import torch

class Settings(BaseSettings):
    # 1. Seguridad: Token de Hugging Face (Protegido como SecretStr)
    hf_token: str = Field(..., validation_alias="HF_TOKEN")
    
    # Parámetros del Modelo y Tarea
    model_id: str = Field(default="bert-base-uncased", validation_alias="MODEL_ID")
    task_name: str = Field(default="mrpc", validation_alias="TASK_NAME")
    task_type: TaskType = Field(default=TaskType.SEQ_CLS, validation_alias="TASK_TYPE")
    max_seq_length: int = Field(default=512, validation_alias="MAX_SEQ_LENGTH")
    output_dir: str = Field(default="./output", validation_alias="OUTPUT_DIR")
    
    # Hiperparámetros de Entrenamiento
    learning_rate: float = Field(default=2e-5, validation_alias="LEARNING_RATE")
    batch_size: int = Field(default=16, validation_alias="BATCH_SIZE")
    epochs: int = Field(default=3, validation_alias="EPOCHS")
    
    # Configuración de LoRA
    execute_lora: bool = Field(default=True, validation_alias="EXECUTE_LORA")
    lora_rank: int = Field(default=8, validation_alias="LORA_RANK")
    lora_alpha: int = Field(default=16, validation_alias="LORA_ALPHA")
    lora_dropout: float = Field(default=0.1, validation_alias="LORA_DROPOUT")
    
    # Configuración de QLoRA
    execute_qlora: bool = Field(default=False, validation_alias="EXECUTE_QLORA")
    qlora_rank: int = Field(default=4, validation_alias="QLORA_RANK")
    qlora_alpha: int = Field(default=8, validation_alias="QLORA_ALPHA")
    qlora_dropout: float = Field(default=0.1, validation_alias="QLORA_DROPOUT")
    
    # Estrategias de Entrenamiento y Guardado
    eval_strategy: str = Field(default="epoch", validation_alias="EVAL_STRATEGY")
    save_strategy: str = Field(default="epoch", validation_alias="SAVE_STRATEGY")
    save_total_limit: int = Field(default=2, validation_alias="SAVE_TOTAL_LIMIT")
    logging_strategy: str = Field(default="epoch", validation_alias="LOGGING_STRATEGY")
    logging_dir: str = Field(default="./logs", validation_alias="LOGGING_DIR")
    
    # Lógica de Ejecución
    use_mock_data: bool = Field(default=False, validation_alias="USE_MOCK_DATA")

    # 2. Portabilidad: Detección Automática de Dispositivo
    @property
    def device(self) -> str:
        """Determina el hardware disponible para ejecución optimizada [6, 7]."""
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    # Configuración del archivo .env
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

# Instancia Global para Inyección de Dependencias
settings = Settings()