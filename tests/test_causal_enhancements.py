"""
Tests for feature-causal-enhancements — Phase 1

Covers:
- AnnealingTemperatureScheduler
- TemperatureSoftmaxAllocationStrategy  (_log_sum_exp helper via strategy)
- CausalMonteCLoRAEngine.allocate_budget() with optional scores dispatch
- CausalMonteCLoRAEngine.warmup() plateau detection
- CausalTrainingConfig new fields and validators
"""
from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.finetuner.causal_engine import (
    AnnealingTemperatureScheduler,
    CausalMonteCLoRAEngine,
    EqualBudgetAllocationStrategy,
    TemperatureSoftmaxAllocationStrategy,
    _log_sum_exp,
)
from src.settings.settings import CausalTrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> CausalMonteCLoRAEngine:
    """Minimal engine with a mocked lora_engine."""
    lora_mock = MagicMock()
    lora_mock.lora_rank = 8
    return CausalMonteCLoRAEngine(
        lora_engine=lora_mock,
        causal_threshold=0.1,
        sample_budget=100,
    )


def _make_linear_model() -> nn.Module:
    """Tiny CPU model with a loss output for warmup tests."""

    class _TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, input_ids, labels=None, **_):
            logits = self.linear(input_ids.float())
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels.long())
            return MagicMock(loss=loss, logits=logits)

    return _TinyClassifier()


def _make_loader(n_batches: int = 5, loss_values: List[float] | None = None):
    """Fake data loader yielding deterministic batches."""

    class _Batch:
        def __init__(self, loss_val):
            self._loss = loss_val
            self.input_ids = torch.zeros(2, 4)
            self.labels = torch.zeros(2, dtype=torch.long)

        def items(self):
            return [("input_ids", self.input_ids), ("labels", self.labels)]

        def __getitem__(self, key):
            return {"input_ids": self.input_ids, "labels": self.labels}[key]

        def keys(self):
            return ["input_ids", "labels"]

    class _Loader:
        def __init__(self):
            self._batches = [
                {"input_ids": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long)}
                for _ in range(n_batches)
            ]

        def __iter__(self):
            return iter(self._batches)

    return _Loader()


# ===========================================================================
# _log_sum_exp
# ===========================================================================

class TestLogSumExp:
    def test_empty_returns_zero(self):
        assert _log_sum_exp([]) == 0.0

    def test_single_element(self):
        assert _log_sum_exp([3.0]) == pytest.approx(3.0, abs=1e-9)

    def test_known_result(self):
        # log(exp(1) + exp(2)) = log(e + e^2)
        expected = math.log(math.exp(1) + math.exp(2))
        assert _log_sum_exp([1.0, 2.0]) == pytest.approx(expected, rel=1e-9)

    def test_large_values_no_overflow(self):
        """Values that would overflow naive exp() are handled correctly."""
        large = [1000.0, 1001.0, 999.0]
        result = _log_sum_exp(large)
        assert math.isfinite(result)
        assert result > 1000.0  # must be > max element

    def test_all_equal_returns_log_n_plus_val(self):
        n, val = 4, 2.0
        expected = val + math.log(n)
        assert _log_sum_exp([val] * n) == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# AnnealingTemperatureScheduler
# ===========================================================================

class TestAnnealingTemperatureScheduler:

    # --- Initialisation validation ---

    def test_rejects_zero_initial_temperature(self):
        with pytest.raises(ValueError, match="initial_temperature"):
            AnnealingTemperatureScheduler(initial_temperature=0.0, enabled=True)

    def test_rejects_negative_initial_temperature(self):
        with pytest.raises(ValueError, match="initial_temperature"):
            AnnealingTemperatureScheduler(initial_temperature=-1.0)

    def test_rejects_zero_temperature_min(self):
        with pytest.raises(ValueError, match="temperature_min"):
            AnnealingTemperatureScheduler(
                initial_temperature=1.0, temperature_min=0.0, enabled=True
            )

    def test_rejects_temperature_min_ge_initial_when_enabled(self):
        with pytest.raises(ValueError, match="temperature_min"):
            AnnealingTemperatureScheduler(
                initial_temperature=0.5, temperature_min=0.5, enabled=True
            )

    def test_allows_temperature_min_ge_initial_when_disabled(self):
        """When enabled=False, the relative ordering constraint is relaxed."""
        sched = AnnealingTemperatureScheduler(
            initial_temperature=0.5, temperature_min=0.8, enabled=False
        )
        assert sched.get_temperature(0) == pytest.approx(0.5)

    def test_rejects_total_steps_zero(self):
        with pytest.raises(ValueError, match="total_steps"):
            AnnealingTemperatureScheduler(initial_temperature=1.0, total_steps=0)

    # --- Disabled mode (no-op) ---

    def test_disabled_always_returns_initial(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=2.0, temperature_min=0.5, total_steps=10, enabled=False
        )
        for step in [0, 5, 10, 100]:
            assert sched.get_temperature(step) == pytest.approx(2.0)

    # --- Enabled mode: linear decay ---

    def test_step_zero_returns_initial(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=2.0, temperature_min=0.2, total_steps=10, enabled=True
        )
        assert sched.get_temperature(0) == pytest.approx(2.0, rel=1e-9)

    def test_step_total_returns_min(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=2.0, temperature_min=0.2, total_steps=10, enabled=True
        )
        assert sched.get_temperature(10) == pytest.approx(0.2, rel=1e-9)

    def test_midpoint_is_average(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=2.0, temperature_min=0.0 + 1e-9,
            total_steps=10, enabled=True,
        )
        # At step 5 of 10, fraction = 0.5; τ = 2.0 - (2.0 - τ_min) * 0.5 ≈ 1.0
        tau = sched.get_temperature(5)
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_beyond_total_steps_clamped_to_min(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=1.0, temperature_min=0.1, total_steps=5, enabled=True
        )
        assert sched.get_temperature(100) == pytest.approx(0.1, rel=1e-9)

    def test_monotonically_decreasing(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=3.0, temperature_min=0.3, total_steps=10, enabled=True
        )
        temps = [sched.get_temperature(s) for s in range(11)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    # --- Properties ---

    def test_properties(self):
        sched = AnnealingTemperatureScheduler(
            initial_temperature=2.0, temperature_min=0.5, total_steps=20, enabled=True
        )
        assert sched.initial_temperature == pytest.approx(2.0)
        assert sched.temperature_min == pytest.approx(0.5)
        assert sched.enabled is True


# ===========================================================================
# TemperatureSoftmaxAllocationStrategy
# ===========================================================================

class TestTemperatureSoftmaxAllocationStrategy:

    # --- Initialisation ---

    def test_rejects_zero_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            TemperatureSoftmaxAllocationStrategy(temperature=0.0)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            TemperatureSoftmaxAllocationStrategy(temperature=-0.5)

    def test_temperature_property_readable(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=2.0)
        assert s.temperature == pytest.approx(2.0)

    def test_temperature_setter_validation(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        with pytest.raises(ValueError):
            s.temperature = 0.0

    def test_temperature_setter_updates_value(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        s.temperature = 0.5
        assert s.temperature == pytest.approx(0.5)

    # --- allocate(): budget sum invariant ---

    def test_budget_sum_equals_total(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        paths = ["a", "b", "c"]
        total = 100
        result = s.allocate(paths, total)
        assert sum(result.values()) == total

    def test_budget_sum_prime_total(self):
        """Prime total exercises largest-remainder rounding."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        paths = ["x", "y", "z"]
        total = 97
        result = s.allocate(paths, total)
        assert sum(result.values()) == total

    def test_single_path_gets_all_budget(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=0.5)
        result = s.allocate(["only"], 42, scores={"only": 5.0})
        assert result == {"only": 42}

    def test_empty_paths_returns_empty(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        assert s.allocate([], 100) == {}

    def test_zero_total_budget_all_zero(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        result = s.allocate(["a", "b"], 0)
        assert all(v == 0 for v in result.values())
        assert sum(result.values()) == 0

    def test_negative_budget_raises(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        with pytest.raises(ValueError, match="total_budget"):
            s.allocate(["a"], -1)

    # --- Score-aware allocation ---

    def test_high_score_path_gets_more_budget_low_temperature(self):
        """Low τ → sharper concentration on high-score paths."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=0.1)
        scores = {"high": 10.0, "low": 0.0}
        result = s.allocate(["high", "low"], 100, scores=scores)
        assert result["high"] > result["low"]

    def test_equal_scores_give_approximately_equal_budget(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        scores = {"a": 1.0, "b": 1.0}
        result = s.allocate(["a", "b"], 100, scores=scores)
        # With equal scores, softmax weights are equal → each gets 50
        assert result["a"] == result["b"] == 50

    def test_no_scores_yields_uniform_distribution(self):
        """Absent scores default to 0.0 → uniform softmax."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        result = s.allocate(["a", "b", "c"], 99)  # 33 each
        assert sum(result.values()) == 99
        # All should be within 1 of each other
        vals = list(result.values())
        assert max(vals) - min(vals) <= 1

    def test_missing_score_path_defaults_to_zero(self):
        """Paths not in scores dict get score 0.0."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=0.1)
        scores = {"a": 5.0}  # "b" not provided
        result = s.allocate(["a", "b"], 100, scores=scores)
        assert sum(result.values()) == 100
        assert result["a"] > result["b"]

    def test_all_keys_present_in_result(self):
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        paths = ["p1", "p2", "p3", "p4"]
        result = s.allocate(paths, 40)
        assert set(result.keys()) == set(paths)

    def test_numerical_stability_extreme_scores(self):
        """Very large score differences must not produce NaN or inf."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        scores = {"dominant": 1000.0, "negligible": -1000.0}
        result = s.allocate(["dominant", "negligible"], 100, scores=scores)
        assert all(math.isfinite(v) for v in result.values())
        assert sum(result.values()) == 100

    def test_protocol_two_arg_call_works(self):
        """Strategy satisfies BudgetAllocationStrategy Protocol (2-arg call)."""
        s = TemperatureSoftmaxAllocationStrategy(temperature=1.0)
        result = s.allocate(["a", "b"], 10)
        assert sum(result.values()) == 10


# ===========================================================================
# CausalMonteCLoRAEngine.allocate_budget() — scores dispatch
# ===========================================================================

class TestAllocateBudgetScoresDispatch:

    def test_equal_strategy_no_scores(self):
        engine = _make_engine()
        result = engine.allocate_budget(["a", "b", "c"], 99)
        assert sum(result.values()) == 99

    def test_equal_strategy_ignores_scores_with_warning(self, caplog):
        engine = _make_engine()
        engine._budget_strategy = EqualBudgetAllocationStrategy()
        with caplog.at_level("WARNING"):
            result = engine.allocate_budget(
                ["a", "b"], 10, scores={"a": 5.0, "b": 1.0}
            )
        assert sum(result.values()) == 10
        assert any("does not accept a 'scores'" in r.message for r in caplog.records)

    def test_temperature_strategy_receives_scores(self):
        lora_mock = MagicMock()
        lora_mock.lora_rank = 8
        engine = CausalMonteCLoRAEngine(
            lora_engine=lora_mock,
            sample_budget=100,
            budget_strategy=TemperatureSoftmaxAllocationStrategy(temperature=0.1),
        )
        scores = {"dominant": 10.0, "weak": 0.0}
        result = engine.allocate_budget(["dominant", "weak"], 100, scores=scores)
        assert result["dominant"] > result["weak"]
        assert sum(result.values()) == 100

    def test_allocation_stored_on_engine(self):
        engine = _make_engine()
        result = engine.allocate_budget(["x", "y"], 20)
        assert engine.budget_allocation == result

    def test_empty_paths_returns_empty(self):
        engine = _make_engine()
        assert engine.allocate_budget([], 100) == {}


# ===========================================================================
# CausalMonteCLoRAEngine.warmup() — plateau detection
# ===========================================================================

class TestWarmupPlateauDetection:

    def test_zero_steps_returns_disabled_state(self):
        engine = _make_engine()
        model = _make_linear_model()
        loader = _make_loader()
        result = engine.warmup(model, loader, num_warmup_steps=0)
        assert result["enabled"] is False
        assert result["early_exit_triggered"] is False
        assert result["plateau_detected_at_step"] is None

    def test_normal_completion_no_plateau(self):
        """When loss consistently decreases, no early exit should occur."""
        engine = _make_engine()
        model = _make_linear_model()
        loader = _make_loader(n_batches=10)
        result = engine.warmup(
            model, loader, num_warmup_steps=5,
            plateau_delta=0.0,   # delta=0 disables plateau effectively
            plateau_patience=100,
        )
        assert result["enabled"] is True
        assert result["completed"] is True
        assert result["early_exit_triggered"] is False
        assert result["plateau_detected_at_step"] is None
        assert result["steps"] == 5

    def test_plateau_triggers_early_exit(self):
        """Force plateau by injecting constant-loss batches."""
        engine = _make_engine()

        # Monkey-patch warmup to use a controlled loss sequence
        call_count = [0]
        constant_losses = [1.0] * 20

        class _ConstantLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, labels=None, **_):
                loss = self.p * 0 + constant_losses[min(call_count[0], len(constant_losses) - 1)]
                call_count[0] += 1
                return MagicMock(loss=loss)

        model = _ConstantLossModel()
        loader = _make_loader(n_batches=20)

        result = engine.warmup(
            model, loader, num_warmup_steps=20,
            plateau_delta=1e-4,
            plateau_patience=3,
        )
        assert result["early_exit_triggered"] is True
        assert result["plateau_detected_at_step"] is not None
        # Should exit after patience (3) steps with no improvement
        assert result["steps"] <= 10  # well before 20

    def test_plateau_detected_at_step_is_correct_step(self):
        """Verify plateau_detected_at_step matches the step count at exit."""
        engine = _make_engine()

        class _ConstantLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, labels=None, **_):
                return MagicMock(loss=self.p * 0 + 1.0)

        model = _ConstantLossModel()
        loader = _make_loader(n_batches=20)

        result = engine.warmup(
            model, loader, num_warmup_steps=20,
            plateau_delta=1e-4,
            plateau_patience=3,
        )
        assert result["early_exit_triggered"] is True
        # plateau_detected_at_step should equal steps at time of exit
        assert result["plateau_detected_at_step"] == result["steps"]

    def test_backward_compatible_keys_present(self):
        """All original warmup() return keys must still be present."""
        engine = _make_engine()
        model = _make_linear_model()
        loader = _make_loader(n_batches=3)
        result = engine.warmup(model, loader, num_warmup_steps=2)
        for key in ("enabled", "completed", "steps", "loss_trajectory"):
            assert key in result, f"Missing backward-compat key: {key}"

    def test_new_diagnostic_keys_present(self):
        """New keys must be present on all warmup return paths."""
        engine = _make_engine()
        model = _make_linear_model()
        loader = _make_loader()
        result = engine.warmup(model, loader, num_warmup_steps=2)
        assert "early_exit_triggered" in result
        assert "plateau_detected_at_step" in result

    def test_loss_trajectory_populated(self):
        engine = _make_engine()
        model = _make_linear_model()
        loader = _make_loader(n_batches=5)
        result = engine.warmup(model, loader, num_warmup_steps=3)
        assert len(result["loss_trajectory"]) == result["steps"]
        assert all(math.isfinite(v) for v in result["loss_trajectory"])


# ===========================================================================
# CausalTrainingConfig — new fields and validators
# ===========================================================================

class TestCausalTrainingConfigNewFields:

    def test_defaults(self):
        cfg = CausalTrainingConfig()
        assert cfg.temperature == pytest.approx(1.0)
        assert cfg.temperature_anneal is False
        assert cfg.temperature_min == pytest.approx(0.1)
        assert cfg.warmup_plateau_delta == pytest.approx(1e-4)
        assert cfg.warmup_plateau_patience == 3

    def test_custom_temperature(self):
        cfg = CausalTrainingConfig(temperature=0.5)
        assert cfg.temperature == pytest.approx(0.5)

    def test_temperature_zero_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CausalTrainingConfig(temperature=0.0)

    def test_temperature_negative_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CausalTrainingConfig(temperature=-1.0)

    def test_temperature_min_zero_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CausalTrainingConfig(temperature_min=0.0)

    def test_warmup_plateau_delta_negative_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CausalTrainingConfig(warmup_plateau_delta=-0.001)

    def test_warmup_plateau_patience_zero_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CausalTrainingConfig(warmup_plateau_patience=0)

    def test_annealing_cross_validator_min_ge_initial_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="temperature_min"):
            CausalTrainingConfig(
                temperature=0.5,
                temperature_min=0.8,
                temperature_anneal=True,
            )

    def test_annealing_cross_validator_min_lt_initial_passes(self):
        cfg = CausalTrainingConfig(
            temperature=1.0,
            temperature_min=0.1,
            temperature_anneal=True,
        )
        assert cfg.temperature_anneal is True

    def test_annealing_disabled_allows_any_min(self):
        """Cross-field validator is skipped when annealing is off."""
        cfg = CausalTrainingConfig(
            temperature=0.5,
            temperature_min=0.8,  # > temperature, but annealing=False
            temperature_anneal=False,
        )
        assert cfg.temperature_min == pytest.approx(0.8)

    def test_existing_fields_unaffected(self):
        """Ensure existing field defaults still work."""
        cfg = CausalTrainingConfig()
        assert cfg.total_causal_budget == 1000
        assert cfg.async_max_steps == 100
        assert cfg.apply_interval == 10
