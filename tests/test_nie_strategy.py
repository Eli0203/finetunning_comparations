"""Unit tests for NIEBudgetAllocationStrategy."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.finetuner.nie_strategy import NIEBudgetAllocationStrategy


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_a = nn.Linear(4, 4)
        self.layer_b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_b(torch.relu(self.layer_a(x)))


def _loader() -> DataLoader:
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_allocate_budget_proportional_to_nie() -> None:
    strategy = NIEBudgetAllocationStrategy(temp_init=2.0, temp_final=0.5)
    paths = ["layer_a", "layer_b"]
    strategy.nie_scores = {"layer_a": 2.0, "layer_b": 0.5}
    strategy._warmup_state["completed"] = True

    allocation = strategy.allocate(paths, 100)

    assert sum(allocation.values()) == 100
    assert allocation["layer_a"] > allocation["layer_b"]


def test_allocate_budget_zero_nie_uniform_fallback_with_remainder_order() -> None:
    strategy = NIEBudgetAllocationStrategy(temp_init=2.0, temp_final=0.5)
    paths = ["p1", "p2", "p3"]
    strategy.nie_scores = {"p1": 0.0, "p2": 0.0, "p3": 0.0}
    strategy._warmup_state["completed"] = True

    allocation = strategy.allocate(paths, 10)

    assert sum(allocation.values()) == 10
    # base=3 remainder=1, stable path order grants remainder to p1
    assert allocation["p1"] == 4
    assert allocation["p2"] == 3
    assert allocation["p3"] == 3


def test_compute_nie_returns_scores_for_requested_paths() -> None:
    strategy = NIEBudgetAllocationStrategy(temp_init=2.0, temp_final=0.5)
    model = TinyModel()
    loader = _loader()
    paths = ["layer_a", "layer_b"]

    scores = strategy.compute_nie(model, loader, paths)

    assert set(scores.keys()) == set(paths)
    assert all(v >= 0.0 for v in scores.values())
