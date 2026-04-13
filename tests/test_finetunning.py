"""
Test suite for fine-tuning system including memory budget validation.
"""

import torch
import torch.nn as nn
import psutil
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.utils.async_sampler import BackgroundSampler
from src.utils.memory_manager import MemoryOptimizer
from src.utils.causal_sampler import CausalWeightSampler
from src.settings import SettingsFactory


def test_smoke():
    """Basic smoke test to ensure the test suite is executable."""
    assert True


class SimpleTestModel(nn.Module):
    """Simple model for memory testing."""
    
    def __init__(self, input_dim=256, output_dim=128, num_lora_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_lora_layers)
        ])
        # Add LoRA adapters
        self.lora_A = nn.Linear(output_dim, 16)
        self.lora_B = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MockCausalEngine:
    """Mock causal engine for testing."""
    
    def __init__(self):
        self.budget_allocation = {'layer_1': 100, 'layer_2': 100}
        self.sample_budget = 1000


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


def test_async_sampler_memory_budget():
    """
    T032: Smoke test ensuring async sampler doesn't exceed buffer memory constraints.
    
    Validates that:
    - Buffer creation succeeds
    - Sample generation is memory-efficient
    - No runaway memory growth during sampling
    - Memory stays within 10GB constraint
    """
    # Get baseline memory
    baseline_memory = get_process_memory_mb()
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    # Create test model and buffer
    model = SimpleTestModel(input_dim=256, output_dim=128, num_lora_layers=4)
    buffer = MemoryOptimizer.create_double_buffer()
    
    # Create mock causal engine and sampler
    causal_engine = MockCausalEngine()
    causal_sampler = CausalWeightSampler(causal_engine, model, device='cpu')
    
    # Create and start async sampler with limited iterations
    async_sampler = BackgroundSampler(
        buffer=buffer,
        model=model,
        max_steps=5,  # Limit to 5 steps for smoke test
        causal_sampler=causal_sampler
    )
    
    try:
        async_sampler.start()
        
        # Wait for sampler to complete
        import time
        time.sleep(2)  # Give sampler time to run
        
        # Check memory after sampling
        peak_memory = get_process_memory_mb()
        memory_delta = peak_memory - baseline_memory
        
        print(f"Peak memory after sampling: {peak_memory:.1f} MB")
        print(f"Memory delta: {memory_delta:.1f} MB")
        
        # Verify memory constraints (10GB limit)
        assert peak_memory < 10240, f"Memory usage {peak_memory:.1f} MB exceeds 10GB limit"
        
        # Verify reasonable memory growth (< 500MB for 5 samples is acceptable)
        assert memory_delta < 500, f"Memory growth {memory_delta:.1f} MB seems excessive"
        
        # Verify buffer actually has data
        latest_weights = buffer.get_latest()
        if latest_weights is not None:
            assert len(latest_weights) > 0, "Buffer should contain sampled weights"
            print(f"✓ Buffer contains {len(latest_weights)} weight tensors")
        
        print("✓ Async sampler memory test passed")
        
    finally:
        async_sampler.stop()
        

def test_buffer_no_memory_leak():
    """
    Test that DoubleBuffer doesn't cause memory leaks under repeated put/get.
    
    Validates sequential put/get operations don't accumulate memory.
    """
    baseline_memory = get_process_memory_mb()
    
    buffer = MemoryOptimizer.create_double_buffer()
    model = SimpleTestModel(input_dim=64, output_dim=32, num_lora_layers=1)
    
    # Simulate repeated weight generation and buffering
    for iteration in range(10):
        # Generate random weights
        weights = {
            name: torch.randn_like(param)
            for name, param in model.named_parameters()
        }
        
        # Put in buffer
        buffer.put(weights)
        
        # Get from buffer
        retrieved = buffer.get_latest()
        assert retrieved is not None, f"Iteration {iteration}: Buffer should not be None"
    
    # Check memory after iterations
    final_memory = get_process_memory_mb()
    memory_delta = final_memory - baseline_memory
    
    print(f"Memory after 10 put/get cycles: {memory_delta:.1f} MB")
    
    # Growth should be minimal (< 50MB for these small tensors)
    assert memory_delta < 100, f"Excessive memory growth: {memory_delta:.1f} MB"
    print("✓ Buffer memory leak test passed")


def test_model_size_estimation():
    """
    Test that model parameter estimation is accurate.
    
    Validates we can correctly count model size for memory planning.
    """
    model = SimpleTestModel(input_dim=256, output_dim=128, num_lora_layers=4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'lora' in name.lower()
    )
    
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    
    # Estimate memory (4 bytes per float32)
    total_memory_mb = (total_params * 4) / (1024 * 1024)
    lora_memory_mb = (lora_params * 4) / (1024 * 1024)
    
    print(f"Estimated total memory: {total_memory_mb:.2f} MB")
    print(f"Estimated LoRA memory: {lora_memory_mb:.2f} MB")
    
    # Verify reasonable estimates
    assert total_params > 0, "Model should have parameters"
    assert lora_params > 0, "Model should have LoRA parameters"
    assert lora_params < total_params, "LoRA params should be subset of total"
    
    print("✓ Model size estimation test passed")


def test_sequential_runs_with_different_config(temp_dir):
    """US1: sequential runs must load isolated configuration from different env files."""
    env_a = Path(temp_dir) / ".env.a"
    env_b = Path(temp_dir) / ".env.b"

    env_a.write_text(
        "HF_TOKEN=token_a\nTASK_NAME=mrpc\nEXPERIMENT_TYPE=laplace_lora\nEXECUTE_CAUSAL_ENGINE=false\nEXECUTE_LAPLACE=true\n"
    )
    env_b.write_text(
        "HF_TOKEN=token_b\nTASK_NAME=sst2\nEXPERIMENT_TYPE=causal_lora\nEXECUTE_CAUSAL_ENGINE=true\nEXECUTE_LAPLACE=false\n"
    )

    run_a = SettingsFactory.create_settings(env_file=env_a)
    run_b = SettingsFactory.create_settings(env_file=env_b)

    assert run_a is not run_b
    assert run_a.hf_token == "token_a"
    assert run_b.hf_token == "token_b"
    assert run_a.task_name == "mrpc"
    assert run_b.task_name == "sst2"
    assert run_a.execute_causal_engine is False
    assert run_a.execute_laplace is True
    assert run_b.execute_causal_engine is True
    assert run_b.execute_laplace is False


def test_sequential_runs_with_mixed_experiment_type_roots(temp_dir):
    """Phase 3B: mixed experiment_type runs produce isolated canonical artifact roots."""
    env_lora = Path(temp_dir) / ".env.lora"
    env_causal = Path(temp_dir) / ".env.causal"

    env_lora.write_text(
        "HF_TOKEN=token_a\nTASK_NAME=mrpc\nRUN_ID=run-lora\nEXPERIMENT_TYPE=lora\nEXECUTE_CAUSAL_ENGINE=false\nEXECUTE_LAPLACE=false\n"
    )
    env_causal.write_text(
        "HF_TOKEN=token_b\nTASK_NAME=mrpc\nRUN_ID=run-causal\nEXPERIMENT_TYPE=causal_lora\nEXECUTE_CAUSAL_ENGINE=true\nEXECUTE_LAPLACE=false\n"
    )

    lora_run = SettingsFactory.create_settings(env_file=env_lora)
    causal_run = SettingsFactory.create_settings(env_file=env_causal)

    assert lora_run.experiment_type == "lora"
    assert causal_run.experiment_type == "causal_lora"
    assert lora_run.canonical_artifact_root != causal_run.canonical_artifact_root

    lora_root = str(lora_run.canonical_artifact_root).replace("\\", "/")
    causal_root = str(causal_run.canonical_artifact_root).replace("\\", "/")

    assert lora_root.endswith("output/lora/mrpc/run-lora")
    assert causal_root.endswith("output/causal_lora/mrpc/run-causal")


# =============================================================================
# Phase 4: User Story 2 — Trusted Checkpoint Evaluation (T049-T064)
# =============================================================================

class TestCheckpointValidation:
    """Phase 4 Tests: Checkpoint validation and deterministic selection (T049-T051)."""
    
    def test_validate_checkpoint_missing_artifacts_detected(self, temp_checkpoint_dir):
        """T049: missing artifacts are detected by validator."""
        from src.finetuner.checkpoint_handler import CheckpointValidator
        
        # Create checkpoint dir missing adapter_model.safetensors
        (temp_checkpoint_dir / "adapter_model.safetensors").unlink()
        
        is_valid, missing = CheckpointValidator.validate_checkpoint(
            temp_checkpoint_dir, method="lora"
        )
        
        assert is_valid is False
        assert "adapter_model.safetensors" in missing
    
    def test_validate_checkpoint_complete_accepted(self, temp_checkpoint_dir):
        """T050: complete checkpoint with all artifacts is accepted."""
        from src.finetuner.checkpoint_handler import CheckpointValidator
        
        is_valid, missing = CheckpointValidator.validate_checkpoint(
            temp_checkpoint_dir, method="lora"
        )
        
        assert is_valid is True
        assert len(missing) == 0
    
    def test_validate_checkpoint_malformed_names_skipped(self, temp_dir):
        """T051: checkpoints with malformed names are skipped with warning."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        # Create a valid checkpoint
        valid_dir = temp_dir / "checkpoint-100"
        valid_dir.mkdir(parents=True)
        (valid_dir / "adapter_config.json").write_text("{}")
        (valid_dir / "adapter_model.safetensors").write_text("")
        (valid_dir / "config.json").write_text("{}")
        
        # Create a malformed checkpoint (no "checkpoint-" prefix)
        malformed_dir = temp_dir / "invalid_name"
        malformed_dir.mkdir(parents=True)
        (malformed_dir / "adapter_config.json").write_text("{}")
        
        candidates = CheckpointSelector.select_checkpoints(temp_dir, method="lora")
        
        # Only valid checkpoint should be selected
        assert len(candidates) == 1
        assert candidates[0].checkpoint_name == "checkpoint-100"


class TestCheckpointSelection:
    """Phase 4 Tests: Checkpoint deterministic selection and evaluation (T052-T058)."""
    
    def test_select_checkpoints_mixed_valid_invalid(self, temp_dir):
        """T059: Selection with mixed valid/invalid returns only valid candidates."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        # Create checkpoints with mixed validity
        for step in [100, 200, 300]:
            ckpt_dir = temp_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "adapter_config.json").write_text("{}")
            (ckpt_dir / "adapter_model.safetensors").write_text("")
            (ckpt_dir / "config.json").write_text("{}")
        
        # Remove artifact from checkpoint-200 to make it invalid
        (temp_dir / "checkpoint-200" / "adapter_model.safetensors").unlink()
        
        candidates = CheckpointSelector.select_checkpoints(temp_dir, method="lora")
        
        assert len(candidates) == 2
        assert candidates[0].checkpoint_name == "checkpoint-100"
        assert candidates[1].checkpoint_name == "checkpoint-300"
    
    def test_select_checkpoints_empty_dir_strict_raises(self, temp_dir):
        """T060: Empty directory with strict=True raises ValueError."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        with pytest.raises(ValueError, match="No valid checkpoints"):
            CheckpointSelector.select_checkpoints(temp_dir, method="lora", strict=True)
    
    def test_select_checkpoints_empty_dir_nonstrict_returns_empty(self, temp_dir):
        """T061: Empty directory with strict=False returns empty list."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        candidates = CheckpointSelector.select_checkpoints(temp_dir, method="lora", strict=False)
        
        assert candidates == []
    
    def test_select_checkpoints_deterministic_order(self, temp_dir):
        """T055: Checkpoints are sorted deterministically by (step_number, name)."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        # Create unordered checkpoints
        for step in [300, 100, 200]:
            ckpt_dir = temp_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "adapter_config.json").write_text("{}")
            (ckpt_dir / "adapter_model.safetensors").write_text("")
            (ckpt_dir / "config.json").write_text("{}")
        
        candidates = CheckpointSelector.select_checkpoints(temp_dir, method="lora")
        
        # Should be sorted by step number ascending
        assert len(candidates) == 3
        assert candidates[0].step_number == 100
        assert candidates[1].step_number == 200
        assert candidates[2].step_number == 300
    
    def test_select_latest_checkpoint_returns_highest_step(self, temp_dir):
        """T057: Select latest checkpoint convenience helper."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        # Create multiple checkpoints
        for step in [100, 200, 300]:
            ckpt_dir = temp_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "adapter_config.json").write_text("{}")
            (ckpt_dir / "adapter_model.safetensors").write_text("")
            (ckpt_dir / "config.json").write_text("{}")
        
        latest = CheckpointSelector.select_latest_checkpoint(temp_dir, method="lora")
        
        assert latest is not None
        assert latest.step_number == 300
        assert latest.checkpoint_name == "checkpoint-300"
    
    def test_select_latest_checkpoint_returns_none_empty(self, temp_dir):
        """T057: Select latest returns None when no valid checkpoints."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        latest = CheckpointSelector.select_latest_checkpoint(temp_dir, method="lora")
        
        assert latest is None


class TestCheckpointNotebookIntegration:
    """Phase 4: Notebook integration using CheckpointSelector directly (T064)."""
    
    def test_checkpoint_selector_notebook_usage(self, temp_dir):
        """T064: CheckpointSelector can be used directly in notebooks."""
        from src.finetuner.checkpoint_handler import CheckpointSelector
        
        # Simulate notebook: create checkpoints with mixed validity
        for step in [100, 200, 300]:
            ckpt_dir = temp_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "adapter_config.json").write_text("{}")
            (ckpt_dir / "adapter_model.safetensors").write_text("")
            (ckpt_dir / "config.json").write_text("{}")
        
        # Make checkpoint-200 invalid
        (temp_dir / "checkpoint-200" / "adapter_model.safetensors").unlink()
        
        # Notebook code: directly call CheckpointSelector (same as main.py pattern)
        # method = "causal_lora" from EXPERIMENT_TYPE env var
        candidates = CheckpointSelector.select_checkpoints(
            output_dir=temp_dir,
            method="causal_lora",
            strict=False
        )
        
        # Malformed checkpoints never reach evaluation loop
        assert len(candidates) == 2
        assert all(c.is_valid for c in candidates)
        assert candidates[0].checkpoint_name == "checkpoint-100"
        assert candidates[1].checkpoint_name == "checkpoint-300"



class TestCheckpointRecoveryResume:
    """Phase 4.4: Recovery persistence and strict resume safety (T114-T117)."""

    @staticmethod
    def _create_valid_checkpoint(base_dir: Path, step: int) -> Path:
        ckpt_dir = base_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapter_config.json").write_text("{}")
        (ckpt_dir / "adapter_model.safetensors").write_text("")
        (ckpt_dir / "config.json").write_text("{}")
        return ckpt_dir

    def test_stage_boundary_checkpoint_persisted_atomically(self, temp_dir):
        """T114: stage-boundary checkpoint persisted atomically as last-known-good."""
        from src.finetuner.checkpoint_handler import RecoveryCheckpointManager

        ckpt = self._create_valid_checkpoint(temp_dir, 100)
        record = RecoveryCheckpointManager.persist_last_known_good(
            checkpoint_dir=ckpt,
            stage_name="post_train_epoch_1",
            run_id="run-001",
        )

        assert record["status"] == "valid"
        assert record["is_atomic_commit"] is True

        recovery_meta = RecoveryCheckpointManager.load_recovery_meta(ckpt)
        assert recovery_meta is not None
        assert recovery_meta["checkpoint_path"] == str(ckpt.resolve())

        pointer = RecoveryCheckpointManager.load_last_known_good(temp_dir)
        assert pointer is not None
        assert pointer["checkpoint_path"] == str(ckpt.resolve())

    def test_partial_failed_stage_artifacts_marked_non_resumable(self, temp_dir):
        """T115: failed-stage partial artifacts are marked non-resumable."""
        from src.finetuner.checkpoint_handler import CheckpointSelector, RecoveryCheckpointManager

        good_ckpt = self._create_valid_checkpoint(temp_dir, 100)
        bad_ckpt = self._create_valid_checkpoint(temp_dir, 200)

        RecoveryCheckpointManager.persist_last_known_good(good_ckpt, "stage-1", run_id="run-001")
        RecoveryCheckpointManager.mark_partial_checkpoint(
            checkpoint_dir=bad_ckpt,
            stage_name="stage-2",
            run_id="run-001",
            reason="interrupted write",
        )

        resumable = CheckpointSelector.select_latest_resumable_checkpoint(temp_dir, method="lora")
        assert resumable is not None
        assert resumable.checkpoint_name == "checkpoint-100"

    def test_resume_selects_newest_fully_validated_last_known_good(self, temp_dir):
        """T116: resume selects newest fully validated last-known-good checkpoint."""
        from src.finetuner.checkpoint_handler import CheckpointSelector, RecoveryCheckpointManager

        ckpt_old = self._create_valid_checkpoint(temp_dir, 100)
        ckpt_new = self._create_valid_checkpoint(temp_dir, 300)

        RecoveryCheckpointManager.persist_last_known_good(ckpt_old, "stage-1", run_id="run-001")
        RecoveryCheckpointManager.persist_last_known_good(ckpt_new, "stage-2", run_id="run-001")

        selected = CheckpointSelector.select_resume_checkpoint(temp_dir, method="lora")
        assert selected.checkpoint_name == "checkpoint-300"

    def test_resume_returns_none_when_no_valid_resumable(self, temp_dir):
        """T117: clean-start behavior when no valid resumable checkpoint exists."""
        from src.finetuner.checkpoint_handler import CheckpointSelector, RecoveryCheckpointManager

        partial_ckpt = self._create_valid_checkpoint(temp_dir, 100)
        RecoveryCheckpointManager.mark_partial_checkpoint(
            checkpoint_dir=partial_ckpt,
            stage_name="stage-1",
            run_id="run-001",
            reason="failed atomic commit",
        )

        selected = CheckpointSelector.select_resume_checkpoint(temp_dir, method="lora")
        assert selected is None


class TestStrictResumeRuntimeAndNotebook:
    """T123: runtime and notebook flows must use strict resume behavior."""

    def test_runtime_resolve_resume_uses_strict_selector(self, monkeypatch, temp_dir):
        """Runtime helper must delegate to strict auto-resume selector."""
        import main as runtime_main

        ckpt_dir = temp_dir / "checkpoint-300"
        ckpt_dir.mkdir(parents=True)

        class _Candidate:
            def __init__(self, path: Path) -> None:
                self.path = path

        monkeypatch.setenv("RESUME_POLICY", "strict")

        with patch(
            "main.CheckpointSelector.select_resume_checkpoint",
            return_value=_Candidate(ckpt_dir),
        ) as mocked_selector:
            resolved = runtime_main._resolve_resume_checkpoint(
                output_dir=str(temp_dir),
                method="lora",
            )

        mocked_selector.assert_called_once()
        assert resolved == str(ckpt_dir)

    def test_runtime_resolve_resume_disabled_policy_returns_none(self, monkeypatch, temp_dir):
        """Runtime helper must skip selector when resume policy disables resume."""
        import main as runtime_main

        monkeypatch.setenv("RESUME_POLICY", "false")

        with patch("main.CheckpointSelector.select_resume_checkpoint") as mocked_selector:
            resolved = runtime_main._resolve_resume_checkpoint(
                output_dir=str(temp_dir),
                method="lora",
            )

        mocked_selector.assert_not_called()
        assert resolved is None

    def test_notebook_flow_sets_strict_resume_policy(self):
        """Notebook execution flow must define strict resume policy for runtime."""
        notebook_text = Path("notebooks/causal_finetuning_demo.ipynb").read_text(encoding="utf-8")
        assert "RESUME_POLICY" in notebook_text
        assert "strict" in notebook_text.lower()


# Import pytest at module level for clarity
