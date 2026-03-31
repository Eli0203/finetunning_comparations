"""AAA-style tests for CausalMonteCLoRAEngine with fixture-based dependency injection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import Mock

import pytest
import torch

from src.finetuner.causal_engine import (
    CausalGradientUnavailableError,
    CausalMonteCLoRAEngine,
    EqualBudgetAllocationStrategy,
)
from src.finetuner.lora_engine import FineTuningEngine


class TinyLossModel(torch.nn.Module):
    """Minimal model returning an object with a loss attribute for causal tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        logits = self.linear(input_ids.float())
        if labels is None:
            return SimpleNamespace(logits=logits)
        target = torch.nn.functional.one_hot(labels, num_classes=2).float()
        loss = torch.nn.functional.mse_loss(logits, target)
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture
def lora_engine_mock() -> Mock:
    """Injected LoRA-engine boundary mock for decoupled causal-engine tests."""
    return Mock(spec=FineTuningEngine)


@pytest.fixture
def tiny_model() -> TinyLossModel:
    """Fixture-provided deterministic tiny model."""
    torch.manual_seed(7)
    return TinyLossModel()


@pytest.fixture
def train_loader() -> List[Dict[str, torch.Tensor]]:
    """Single-batch fixture to keep tests fast and deterministic."""
    return [
        {
            "input_ids": torch.ones((3, 4), dtype=torch.float32),
            "attention_mask": torch.ones((3, 4), dtype=torch.int64),
            "label": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    ]


def test_engine_uses_injected_dependencies(lora_engine_mock: Mock) -> None:
    """Engine should compose injected collaborators and defaults."""
    # Arrange
    engine = CausalMonteCLoRAEngine(lora_engine=lora_engine_mock)

    # Act
    summary = engine.get_causal_summary()

    # Assert
    assert engine.lora_engine is lora_engine_mock
    assert isinstance(engine._budget_strategy, EqualBudgetAllocationStrategy)
    assert summary["causal_paths"] == []


def test_identify_causal_paths_empty_loader_raises(
    lora_engine_mock: Mock,
    tiny_model: TinyLossModel,
) -> None:
    """Empty loaders must fail loudly to avoid hidden baseline fallback behavior."""
    # Arrange
    engine = CausalMonteCLoRAEngine(lora_engine=lora_engine_mock)

    # Act / Assert
    with pytest.raises(CausalGradientUnavailableError):
        engine.identify_causal_paths(tiny_model, [])


def test_filter_model_inputs_maps_label_and_drops_unknowns() -> None:
    """Input filtering should normalize labels and remove unsupported keys."""
    # Arrange
    batch = {
        "input_ids": torch.ones((2, 4)),
        "attention_mask": torch.ones((2, 4)),
        "label": torch.tensor([0, 1]),
        "metadata": "ignore-me",
    }

    # Act
    filtered = CausalMonteCLoRAEngine._filter_model_inputs(batch)

    # Assert
    assert "labels" in filtered
    assert "label" not in filtered
    assert "metadata" not in filtered


def test_allocate_budget_uses_strategy(lora_engine_mock: Mock) -> None:
    """Budget distribution must delegate to the injected strategy object."""
    # Arrange
    strategy = Mock()
    strategy.allocate.return_value = {"a": 7, "b": 3}
    engine = CausalMonteCLoRAEngine(lora_engine=lora_engine_mock, budget_strategy=strategy)

    # Act
    allocation = engine.allocate_budget(["a", "b"], 10)

    # Assert
    strategy.allocate.assert_called_once_with(["a", "b"], 10)
    assert allocation == {"a": 7, "b": 3}


def test_equal_strategy_distributes_remainder() -> None:
    """Even strategy should preserve totals and spread the remainder across early paths."""
    # Arrange
    strategy = EqualBudgetAllocationStrategy()

    # Act
    allocation = strategy.allocate(["p1", "p2", "p3"], 10)

    # Assert
    assert allocation == {"p1": 4, "p2": 3, "p3": 3}
    assert sum(allocation.values()) == 10


def test_identify_causal_paths_happy_path(
    lora_engine_mock: Mock,
    tiny_model: TinyLossModel,
    train_loader: List[Dict[str, torch.Tensor]],
) -> None:
    """A valid gradient pass should identify at least one sensitive module."""
    # Arrange
    engine = CausalMonteCLoRAEngine(
        lora_engine=lora_engine_mock,
        causal_threshold=0.0,
    )

    # Act
    paths = engine.identify_causal_paths(tiny_model, train_loader)

    # Assert
    assert len(paths) >= 1
    assert any("linear" in name for name in paths)
