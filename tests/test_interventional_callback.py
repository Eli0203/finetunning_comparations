"""US3 tests for InterventionalWeightCallback."""

import tempfile

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.finetuner.causal_training_orchestrator import InterventionalWeightCallback


def _hf_args() -> TrainingArguments:
    return TrainingArguments(output_dir=tempfile.mkdtemp())


def test_interventional_weight_computation_adds_weights() -> None:
    callback = InterventionalWeightCallback(window_size=10, max_weight=10.0)
    args = _hf_args()
    state = TrainerState(global_step=1)
    control = TrainerControl()
    inputs = {
        'input_ids': torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=torch.long)
    }

    callback.on_step_begin(args, state, control, inputs=inputs)

    assert 'interventional_weights' in inputs
    assert inputs['interventional_weights'].shape[0] == 3


def test_interventional_weights_are_clamped_and_finite() -> None:
    callback = InterventionalWeightCallback(window_size=2, max_weight=2.5)
    args = _hf_args()
    control = TrainerControl()

    # Repeated rare patterns should still produce bounded finite weights.
    for step in range(1, 6):
        inputs = {'input_ids': torch.tensor([[step, step + 1, step + 2]], dtype=torch.long)}
        state = TrainerState(global_step=step)
        callback.on_step_begin(args, state, control, inputs=inputs)
        w = inputs['interventional_weights']
        assert torch.isfinite(w).all()
        assert float(w.max()) <= 2.5 + 1e-6


def test_interventional_callback_metrics_reporting() -> None:
    callback = InterventionalWeightCallback(window_size=5, max_weight=10.0)
    args = _hf_args()
    state = TrainerState(global_step=1)
    control = TrainerControl()
    inputs = {'input_ids': torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long)}

    callback.on_step_begin(args, state, control, inputs=inputs)
    metrics = callback.get_metrics()

    assert metrics['applied_steps'] == 1
    assert metrics['window_size'] == 5
    assert metrics['tracked_features'] >= 1
    assert metrics['last_mean_weight'] > 0
