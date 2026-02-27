"""
Finetuning Engine for LoRA-based Adaptation
Author: Eliana Vallejo
"""

from typing import Protocol, List
from peft import get_peft_model, LoraConfig, TaskType
from src.utils.logger import logger

class ModelConfig(Protocol):
    """Protocol for model abstraction to ensure loose coupling [2]."""
    model_name: str

class FineTuningEngine:
    # FIXED: Constructor now accepts the 5 arguments passed in main.py
    def __init__(self, task_type: TaskType, model, rank: int, alpha: int, dropout: float):
        self.model = model
        logger.debug(f"Engine initialized with model type: {type(model)}")
        
        # 3.1 LOW-RANK ADAPTATION (LORA) logic [3, 4]
        # Parameters are now injected instead of hardcoded
        self._config = LoraConfig(
            task_type=task_type, 
            r=rank,           # Rank of decomposition (e.g., 8) [4]
            lora_alpha=alpha, # Scaling factor (e.g., 16) [5]
            target_modules=["query", "value"], # Target attention layers [6]
            lora_dropout=dropout
        )

    def apply_lora(self):
        """Wraps the model with LoRA layers to enable parameter-efficient fine-tuning [7, 8]."""
        logger.info(f"Applying LoRA (r={self._config.r}, alpha={self._config.lora_alpha}) to base model...")
        
        # get_peft_model creates the adapter matrices A and B [4, 9]
        self.peft_model = get_peft_model(self.model, self._config)
        
        # Traceability: Calculate trainable parameters to ensure efficiency [4, 8]
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,}")
        
        return self.peft_model