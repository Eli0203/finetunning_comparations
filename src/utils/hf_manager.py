"""
Hugging Face Dataset Manager for GLUE tasks with caching and authentication
Author: Eliana Vallejo
"""
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

from src.utils.logger import logger

class HFSettings(BaseSettings):
    hf_token: str
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

class HFDatasetManager:
    """Singleton-style manager for Hugging Face interactions."""
    def __init__(self, settings: HFSettings):
        self._cache: Dict[str, DatasetDict] = {}
        login(token=settings.hf_token)

    def get_glue_task(self, task_name: str):
        if task_name not in self._cache:
            logger.info(f"Downloading GLUE task: {task_name}") 
            self._cache[task_name] = load_dataset("glue", task_name)
            logger.debug(f"Task {task_name} cached successfully.")
        return self._cache[task_name]

# Single global instance
hf_client = HFDatasetManager(HFSettings())