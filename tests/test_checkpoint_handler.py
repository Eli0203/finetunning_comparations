from pathlib import Path

from src.finetuner.checkpoint_handler import (
    CheckpointSelector,
    CheckpointValidator,
    RecoveryCheckpointManager,
)


def _create_lora_checkpoint(base_dir: Path, step: int) -> Path:
    ckpt = base_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_text("", encoding="utf-8")
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    return ckpt


def test_laplace_requires_posterior_artifacts(temp_dir):
    ckpt = _create_lora_checkpoint(temp_dir, 100)

    is_valid, missing = CheckpointValidator.validate_checkpoint(ckpt, method="laplace_lora")

    assert is_valid is False
    assert "posterior_mean.pt" in missing
    assert "posterior_cov.pt" in missing


def test_partial_checkpoint_is_skipped_for_resumable_selection(temp_dir):
    good_ckpt = _create_lora_checkpoint(temp_dir, 100)
    partial_ckpt = _create_lora_checkpoint(temp_dir, 200)

    RecoveryCheckpointManager.persist_last_known_good(good_ckpt, "stage-1", run_id="run-001")
    RecoveryCheckpointManager.mark_partial_checkpoint(
        partial_ckpt,
        "stage-2",
        run_id="run-001",
        reason="interrupted",
    )

    selected = CheckpointSelector.select_latest_resumable_checkpoint(temp_dir, method="lora")

    assert selected is not None
    assert selected.checkpoint_name == "checkpoint-100"


def test_resume_checkpoint_falls_back_to_clean_start(temp_dir):
    partial_ckpt = _create_lora_checkpoint(temp_dir, 100)
    RecoveryCheckpointManager.mark_partial_checkpoint(
        partial_ckpt,
        "stage-1",
        run_id="run-001",
        reason="failed-atomic-commit",
    )

    selected = CheckpointSelector.select_resume_checkpoint(temp_dir, method="lora")

    assert selected is None
    assert CheckpointSelector.select_latest_resumable_checkpoint(temp_dir, method="lora") is None
    assert RecoveryCheckpointManager.load_last_known_good(temp_dir) is None
