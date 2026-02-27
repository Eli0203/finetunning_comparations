"""
Data loader for GLUE tasks using Hugging Face Datasets and Transformers
Author: Eliana Vallejo
"""

from posixpath import split
from typing import Protocol, Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.utils.hf_manager import hf_client

class GLUEConfig(Protocol):
    task_name: str
    batch_size: int
    max_length: int

class GLUEDataLoader:
    def __init__(self, model_id: str, task_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use the singleton client to retrieve data
        self.dataset = hf_client.get_glue_task(task_name)

    def _tokenize_fn(self, examples):
        return self.tokenizer(
            examples["sentence1"], 
            examples["sentence2"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    def get_loader(self, split: str, batch_size: int = 8) -> DataLoader:
        tokenized = self.dataset[split].map(self._tokenize_fn, batched=True)
    
    #Error: ValueError: You must specify exactly one of input_ids or inputs_embeds REQUIRED FIX: Explicitly set format for BERT forward pass arguments
        tokenized.set_format(
        type="torch", 
        columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    
        return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"))
    
    def get_loaders(self) -> Dict[str, DataLoader]:
        tokenized_datasets = self.raw_datasets.map(
            self._tokenize_fn, batched=True
        )
        # Use PyTorch DataLoader for batch management 
        loaders = {
            split: DataLoader(
                tokenized_datasets[split], 
                batch_size=self.config.batch_size, 
                shuffle=(split == "train")
            )
            for split in ["train", "validation", "test"]
        }
        return loaders