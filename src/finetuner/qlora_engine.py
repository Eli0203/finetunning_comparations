import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class QLoraEngine:
    def __init__(self, model_id: str, rank: int, alpha: int, dropout: float):
        self.model_id = model_id
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) 

    def prepare_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto"
        ) 
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=["query", "value"],
            task_type="SEQ_CLS"
        ) 
        
        return get_peft_model(model, lora_config)