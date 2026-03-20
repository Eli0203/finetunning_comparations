"""
Data loader for GLUE tasks using Hugging Face Datasets and Transformers
Author: Eliana Vallejo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils.hf_manager import hf_client


class GLUEConfig(Protocol):
    task_name: str
    batch_size: int
    max_seq_length: int


class GLUETaskStrategy(Protocol):
    """Task-specific tokenization strategy contract for GLUE datasets."""

    task_name: str
    text_keys: Tuple[str, Optional[str]]
    default_eval_split: str

    def tokenize_batch(self, examples: Dict[str, Sequence[str]], tokenizer, max_length: int) -> Dict[str, List[int]]:
        """Tokenize a batch of examples for this task."""

    def format_columns(self, dataset_columns: Sequence[str]) -> List[str]:
        """Return model-ready tensor columns for set_format()."""


@dataclass(frozen=True)
class BaseGLUEStrategy:
    """Reusable base implementation for GLUE tokenization strategies."""

    task_name: str
    text_keys: Tuple[str, Optional[str]]
    default_eval_split: str = "validation"

    def tokenize_batch(self, examples: Dict[str, Sequence[str]], tokenizer, max_length: int) -> Dict[str, List[int]]:
        key_a, key_b = self.text_keys
        if key_b is None:
            return tokenizer(
                examples[key_a],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            examples[key_a],
            examples[key_b],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def format_columns(self, dataset_columns: Sequence[str]) -> List[str]:
        # Keep the minimal input set to reduce tensor memory footprint.
        columns = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in dataset_columns:
            columns.insert(1, "token_type_ids")
        return columns


class MRPCStrategy(BaseGLUEStrategy):
    def __init__(self) -> None:
        super().__init__(task_name="mrpc", text_keys=("sentence1", "sentence2"), default_eval_split="validation")


class SST2Strategy(BaseGLUEStrategy):
    def __init__(self) -> None:
        super().__init__(task_name="sst2", text_keys=("sentence", None), default_eval_split="validation")


class QNLIStrategy(BaseGLUEStrategy):
    def __init__(self) -> None:
        super().__init__(task_name="qnli", text_keys=("question", "sentence"), default_eval_split="validation")


class GLUETaskStrategyFactory:
    """Factory for selecting task-specific GLUE tokenization strategies."""

    _strategies: Dict[str, GLUETaskStrategy] = {
        "mrpc": MRPCStrategy(),
        "sst2": SST2Strategy(),
        "qnli": QNLIStrategy(),
    }

    @classmethod
    def create(cls, task_name: str) -> GLUETaskStrategy:
        key = task_name.lower().strip()
        strategy = cls._strategies.get(key)
        if strategy is None:
            supported = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(f"Unsupported GLUE task '{task_name}'. Supported tasks: {supported}")
        return strategy

    @classmethod
    def supported_tasks(cls) -> List[str]:
        return sorted(cls._strategies.keys())


class GLUEDataLoader:
    """Load, tokenize, and format GLUE datasets using a task strategy."""

    def __init__(self, model_id: str, task_name: str, max_length: int = 128):
        """Initialize tokenizer, task strategy, and raw dataset.

        Args:
            model_id: Hugging Face model identifier.
            task_name: GLUE task name (mrpc, sst2, qnli).
            max_length: Max sequence length used during tokenization.
        """
        self.task_name = task_name.lower().strip()
        self.max_length = max_length
        self.strategy = GLUETaskStrategyFactory.create(self.task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset = hf_client.get_glue_task(task_name)
        self._tokenized_cache: Dict[str, Dataset] = {}

    @staticmethod
    def supported_tasks() -> List[str]:
        """Return supported GLUE task names."""
        return GLUETaskStrategyFactory.supported_tasks()

    def _tokenize_fn(self, examples):
        """Backward-compatible wrapper; use strategy tokenization internally."""
        return self.strategy.tokenize_batch(examples, self.tokenizer, self.max_length)

    def _resolve_split(self, split: str) -> str:
        if split in self.dataset:
            return split
        if split == "validation" and "validation_matched" in self.dataset:
            return "validation_matched"
        available = ", ".join(self.dataset.keys())
        raise ValueError(f"Split '{split}' is not available for task '{self.task_name}'. Available: {available}")

    def get_dataset(self, split: str) -> Dataset:
        """Return tokenized+formatted dataset split ready for Trainer/DataLoader."""
        resolved_split = self._resolve_split(split)
        cached = self._tokenized_cache.get(resolved_split)
        if cached is not None:
            return cached

        tokenized = self.dataset[resolved_split].map(
            self._tokenize_fn,
            batched=True,
        )
        tokenized.set_format(
            type="torch",
            columns=self.strategy.format_columns(tokenized.column_names),
        )
        self._tokenized_cache[resolved_split] = tokenized
        return tokenized

    def get_loader(self, split: str, batch_size: int = 8) -> DataLoader:
        """Return a PyTorch DataLoader for a tokenized GLUE split."""
        tokenized = self.get_dataset(split)
        return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"))

    def get_datasets(
        self,
        train_split: str = "train",
        eval_split: Optional[str] = None,
    ) -> Tuple[Dataset, Dataset, AutoTokenizer]:
        """Return train/eval datasets plus tokenizer for training workflows."""
        target_eval_split = eval_split or self.strategy.default_eval_split
        train_ds = self.get_dataset(train_split)
        eval_ds = self.get_dataset(target_eval_split)
        return train_ds, eval_ds, self.tokenizer

    def get_loaders(self) -> Dict[str, DataLoader]:
        """Backward-compatible helper returning train/eval DataLoaders."""
        train_loader = self.get_loader("train")
        eval_loader = self.get_loader(self.strategy.default_eval_split)
        loaders = {"train": train_loader, "validation": eval_loader}
        if "test" in self.dataset:
            loaders["test"] = self.get_loader("test")
        return loaders