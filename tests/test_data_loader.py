"""Tests for Strategy-based GLUEDataLoader."""

from __future__ import annotations

from typing import List

from datasets import Dataset, DatasetDict

from src.finetuner.data_loader import GLUEDataLoader


class FakeTokenizer:
    """Lightweight tokenizer stub for deterministic tests."""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def __call__(self, text_a, text_b=None, truncation=True, padding="max_length", max_length=128):
        batch_size = len(text_a)
        self.calls.append(
            {
                "has_pair": text_b is not None,
                "batch_size": batch_size,
                "max_length": max_length,
                "truncation": truncation,
                "padding": padding,
            }
        )
        return {
            "input_ids": [[1] * max_length for _ in range(batch_size)],
            "attention_mask": [[1] * max_length for _ in range(batch_size)],
            "token_type_ids": [[0] * max_length for _ in range(batch_size)],
        }


def _build_task_dataset(task_name: str) -> DatasetDict:
    if task_name == "mrpc":
        train = Dataset.from_dict(
            {
                "sentence1": ["a", "b"],
                "sentence2": ["c", "d"],
                "label": [1, 0],
            }
        )
        val = Dataset.from_dict(
            {
                "sentence1": ["e"],
                "sentence2": ["f"],
                "label": [1],
            }
        )
    elif task_name == "sst2":
        train = Dataset.from_dict(
            {
                "sentence": ["good", "bad"],
                "label": [1, 0],
            }
        )
        val = Dataset.from_dict(
            {
                "sentence": ["ok"],
                "label": [1],
            }
        )
    else:  # qnli
        train = Dataset.from_dict(
            {
                "question": ["q1", "q2"],
                "sentence": ["s1", "s2"],
                "label": [0, 1],
            }
        )
        val = Dataset.from_dict(
            {
                "question": ["q3"],
                "sentence": ["s3"],
                "label": [1],
            }
        )

    return DatasetDict({"train": train, "validation": val})


def _patch_dependencies(monkeypatch, tokenizer: FakeTokenizer):
    monkeypatch.setattr(
        "src.finetuner.data_loader.AutoTokenizer.from_pretrained",
        lambda _model_id: tokenizer,
    )
    monkeypatch.setattr(
        "src.finetuner.data_loader.hf_client.get_glue_task",
        lambda task_name: _build_task_dataset(task_name),
    )


def test_supported_tasks():
    assert GLUEDataLoader.supported_tasks() == ["mrpc", "qnli", "sst2"]


def test_mrpc_uses_pair_strategy(monkeypatch):
    tokenizer = FakeTokenizer()
    _patch_dependencies(monkeypatch, tokenizer)

    loader = GLUEDataLoader(model_id="bert-base-uncased", task_name="mrpc", max_length=16)
    train_ds, eval_ds, _ = loader.get_datasets()

    assert len(train_ds) == 2
    assert len(eval_ds) == 1
    assert tokenizer.calls
    assert tokenizer.calls[0]["has_pair"] is True


def test_sst2_uses_single_sentence_strategy(monkeypatch):
    tokenizer = FakeTokenizer()
    _patch_dependencies(monkeypatch, tokenizer)

    loader = GLUEDataLoader(model_id="bert-base-uncased", task_name="sst2", max_length=8)
    train_ds, _, _ = loader.get_datasets()

    assert len(train_ds) == 2
    assert tokenizer.calls[0]["has_pair"] is False


def test_qnli_get_loader(monkeypatch):
    tokenizer = FakeTokenizer()
    _patch_dependencies(monkeypatch, tokenizer)

    loader = GLUEDataLoader(model_id="bert-base-uncased", task_name="qnli", max_length=8)
    data_loader = loader.get_loader("train", batch_size=2)

    batch = next(iter(data_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
