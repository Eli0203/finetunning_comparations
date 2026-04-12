"""
Checkpoint validation and deterministic selection for model evaluation.

This module provides utilities to validate checkpoint directory structures,
detect malformed checkpoints, and select valid checkpoints in deterministic order
for reproducible evaluation across different experiment types (LoRA, Laplace-LoRA, Causal-LoRA).

Core Concepts:
- Validation: Ensures required artifacts are present for each experiment type
- Selection: Skips malformed checkpoints with warnings; fails only if strict and none remain
- Determinism: Sorts by (step_number, checkpoint_name) for reproducible test order
"""

from pathlib import Path
from typing import List, Tuple
import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _build_structured_diagnostic(
    error_type: str,
    level: str,
    context: dict,
    message: str,
    resolution: str | None = None,
) -> dict:
    """Build canonical structured diagnostic payload for warnings/errors."""
    payload = {
        "error_type": error_type,
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "message": message,
    }
    if resolution is not None:
        payload["resolution"] = resolution
    return payload


def _atomic_write_json(file_path: Path, payload: dict) -> None:
    """Atomically write JSON to avoid partial metadata visibility."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(file_path)


class RecoveryCheckpointManager:
    """Persist and read last-known-good recovery metadata for checkpoints."""

    RECOVERY_META_FILENAME = "recovery_meta.json"
    LAST_KNOWN_GOOD_FILENAME = "last_known_good.json"

    @staticmethod
    def _recovery_meta_path(checkpoint_dir: Path) -> Path:
        return checkpoint_dir / RecoveryCheckpointManager.RECOVERY_META_FILENAME

    @staticmethod
    def _last_known_good_path(output_dir: Path) -> Path:
        return output_dir / RecoveryCheckpointManager.LAST_KNOWN_GOOD_FILENAME

    @staticmethod
    def persist_last_known_good(
        checkpoint_dir: Path,
        stage_name: str,
        run_id: str | None = None,
    ) -> dict:
        """Persist an atomic, resumable checkpoint marker for successful stage boundary."""
        checkpoint_dir = checkpoint_dir.resolve()
        output_dir = checkpoint_dir.parent.resolve()
        record = {
            "run_id": run_id,
            "stage_name": stage_name,
            "checkpoint_path": str(checkpoint_dir),
            "status": "valid",
            "is_atomic_commit": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(RecoveryCheckpointManager._recovery_meta_path(checkpoint_dir), record)
        _atomic_write_json(RecoveryCheckpointManager._last_known_good_path(output_dir), record)
        return record

    @staticmethod
    def mark_partial_checkpoint(
        checkpoint_dir: Path,
        stage_name: str,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        """Mark failed-stage checkpoint artifacts as non-resumable (partial)."""
        checkpoint_dir = checkpoint_dir.resolve()
        record = {
            "run_id": run_id,
            "stage_name": stage_name,
            "checkpoint_path": str(checkpoint_dir),
            "status": "partial",
            "is_atomic_commit": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        _atomic_write_json(RecoveryCheckpointManager._recovery_meta_path(checkpoint_dir), record)
        return record

    @staticmethod
    def load_recovery_meta(checkpoint_dir: Path) -> dict | None:
        """Load per-checkpoint recovery metadata if present."""
        meta_path = RecoveryCheckpointManager._recovery_meta_path(checkpoint_dir)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    @staticmethod
    def load_last_known_good(output_dir: Path) -> dict | None:
        """Load run-level last-known-good pointer if present."""
        pointer_path = RecoveryCheckpointManager._last_known_good_path(output_dir)
        if not pointer_path.exists():
            return None
        return json.loads(pointer_path.read_text(encoding="utf-8"))


class CheckpointCandidate:
    """
    Represents a potential checkpoint location with validation status.
    
    Attributes:
        path: Absolute path to checkpoint directory
        step_number: Extracted from "checkpoint-{step}"
        checkpoint_name: e.g., "checkpoint-345"
        is_valid: All required artifacts present?
        artifacts_missing: List of missing artifact names
    """
    
    def __init__(
        self,
        path: Path,
        step_number: int,
        checkpoint_name: str,
        is_valid: bool,
        artifacts_missing: List[str] | None = None,
    ):
        self.path = path
        self.step_number = step_number
        self.checkpoint_name = checkpoint_name
        self.is_valid = is_valid
        self.artifacts_missing = artifacts_missing or []
    
    @property
    def sort_key(self) -> Tuple[int, str]:
        """Return deterministic sort key: (step_number, checkpoint_name)."""
        return (self.step_number, self.checkpoint_name)
    
    def __repr__(self) -> str:
        return (
            f"CheckpointCandidate(path={self.path.name}, step={self.step_number}, "
            f"valid={self.is_valid})"
        )


class CheckpointValidator:
    """Validate checkpoint directory structure and required artifacts."""
    
    # Required artifacts by experiment method
    REQUIRED_ARTIFACTS_LORA = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
    }
    
    REQUIRED_ARTIFACTS_LAPLACE = REQUIRED_ARTIFACTS_LORA | {
        "posterior_mean.pt",
        "posterior_cov.pt",
    }
    
    REQUIRED_ARTIFACTS_CAUSAL = REQUIRED_ARTIFACTS_LORA  # Same as LoRA for now

    @staticmethod
    def validate_checkpoint(
        checkpoint_dir: Path,
        method: str = "lora",
    ) -> Tuple[bool, List[str]]:
        """
        Validate checkpoint directory has all required artifacts.
        
        Args:
            checkpoint_dir: Path to checkpoint-{N} directory
            method: "lora", "laplace", or "causal"
        
        Returns:
            (is_valid, missing_files) tuple where:
            - is_valid: True if all required artifacts present
            - missing_files: List of missing artifact names (empty if valid)
        
        Raises:
            FileNotFoundError: If checkpoint_dir does not exist
        """
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Select required artifacts based on method
        if method == "lora":
            required = CheckpointValidator.REQUIRED_ARTIFACTS_LORA
        elif method == "laplace" or method == "laplace_lora":
            required = CheckpointValidator.REQUIRED_ARTIFACTS_LAPLACE
        elif method == "causal" or method == "causal_lora":
            required = CheckpointValidator.REQUIRED_ARTIFACTS_CAUSAL
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check each artifact
        missing = []
        for artifact in required:
            artifact_path = checkpoint_dir / artifact
            if not artifact_path.exists():
                missing.append(artifact)
        
        is_valid = len(missing) == 0
        return is_valid, missing


class CheckpointSelector:
    """Select valid checkpoints deterministically; skip malformed ones."""
    
    @staticmethod
    def select_checkpoints(
        output_dir: Path,
        method: str = "lora",
        select_top_n: int | None = None,
        strict: bool = False,
    ) -> List[CheckpointCandidate]:
        """
        Select valid checkpoints from output directory.
        
        Algorithm:
        1. Glob checkpoint-* directories from output_dir
        2. For each candidate:
           a. Parse step_number from name (e.g., checkpoint-345 → 345)
           b. Validate artifacts present
           c. If valid: add to candidates; if invalid: log warning, skip
        3. Sort candidates deterministically by (step_number, name)
        4. If strict and no candidates: raise ValueError
        5. Return top N if select_top_n set, else all
        
        Args:
            output_dir: Training output directory (contains checkpoint-* subdirs)
            method: "lora", "laplace", "laplace_lora", "causal" or "causal_lora"
            select_top_n: If set, return top N by step number (most recent)
            strict: If True, fail if zero valid checkpoints; if False, return []
        
        Returns:
            List[CheckpointCandidate] sorted deterministically (ascending step_number)
        
        Raises:
            FileNotFoundError: If output_dir does not exist
            ValueError: If strict=True and zero valid checkpoints remain
        """
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
        # Get required artifacts for validation
        if method in ("laplace", "laplace_lora"):
            required_artifacts = CheckpointValidator.REQUIRED_ARTIFACTS_LAPLACE
        elif method in ("causal", "causal_lora"):
            required_artifacts = CheckpointValidator.REQUIRED_ARTIFACTS_CAUSAL
        else:
            required_artifacts = CheckpointValidator.REQUIRED_ARTIFACTS_LORA
        
        # Glob all checkpoint-* directories
        checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
        
        valid_candidates = []
        
        for checkpoint_dir in checkpoint_dirs:
            # Parse step number from directory name
            try:
                # Extract numeric part from "checkpoint-{N}"
                step_str = checkpoint_dir.name.split("-")[-1]
                step_number = int(step_str)
            except (ValueError, IndexError):
                logger.warning(
                    f"Could not parse step number from {checkpoint_dir.name}; skipping"
                )
                continue
            
            # Validate artifacts
            try:
                is_valid, missing = CheckpointValidator.validate_checkpoint(
                    checkpoint_dir, method
                )
            except FileNotFoundError as e:
                logger.warning(f"Could not validate {checkpoint_dir.name}: {e}; skipping")
                continue
            
            if not is_valid:
                logger.warning(
                    f"Checkpoint {checkpoint_dir.name} missing artifacts: {missing}; skipping"
                )
                continue
            
            # Valid candidate
            candidate = CheckpointCandidate(
                path=checkpoint_dir,
                step_number=step_number,
                checkpoint_name=checkpoint_dir.name,
                is_valid=True,
                artifacts_missing=[],
            )
            valid_candidates.append(candidate)
        
        # Fail if none valid and strict mode
        if len(valid_candidates) == 0 and strict:
            raise ValueError(
                f"No valid checkpoints found in {output_dir} for method={method}. "
                f"Required artifacts: {required_artifacts}"
            )
        
        # Sort deterministically by (step_number, checkpoint_name)
        valid_candidates.sort(key=lambda c: c.sort_key)
        
        # Optionally return top N (most recent)
        if select_top_n is not None:
            valid_candidates = valid_candidates[-select_top_n:]
        
        return valid_candidates
    
    @staticmethod
    def select_latest_checkpoint(
        output_dir: Path,
        method: str = "lora",
    ) -> CheckpointCandidate | None:
        """
        Select single highest-step checkpoint (most recent).
        
        Returns None if no valid checkpoints.
        
        Args:
            output_dir: Training output directory
            method: "lora", "laplace", "laplace_lora", "causal" or "causal_lora"
        
        Returns:
            CheckpointCandidate for latest valid checkpoint, or None if none found
        """
        try:
            candidates = CheckpointSelector.select_checkpoints(
                output_dir, method=method, select_top_n=1, strict=False
            )
            return candidates[-1] if candidates else None
        except FileNotFoundError:
            return None

    @staticmethod
    def select_latest_resumable_checkpoint(
        output_dir: Path,
        method: str = "lora",
    ) -> CheckpointCandidate | None:
        """Return newest checkpoint eligible for resume (valid + atomic + non-partial)."""
        candidates = CheckpointSelector.select_checkpoints(
            output_dir=output_dir,
            method=method,
            strict=False,
        )

        resumable: List[CheckpointCandidate] = []
        for candidate in candidates:
            recovery_meta = RecoveryCheckpointManager.load_recovery_meta(candidate.path)
            if recovery_meta is None:
                continue
            if recovery_meta.get("status") != "valid":
                continue
            if recovery_meta.get("is_atomic_commit") is not True:
                continue
            resumable.append(candidate)

        if not resumable:
            return None

        resumable.sort(key=lambda c: c.sort_key)
        return resumable[-1]

    @staticmethod
    def select_resume_checkpoint(
        output_dir: Path,
        method: str = "lora",
    ) -> CheckpointCandidate:
        """Select newest fully validated last-known-good checkpoint for strict auto-resume.

        Raises:
            RuntimeError: When no valid resumable checkpoint exists. Exception message
                is a JSON structured diagnostic payload.
        """
        output_dir = output_dir.resolve()
        selected = CheckpointSelector.select_latest_resumable_checkpoint(
            output_dir=output_dir,
            method=method,
        )
        if selected is None:
            diagnostic = _build_structured_diagnostic(
                error_type="resume_no_valid_checkpoint",
                level="error",
                context={
                    "output_dir": str(output_dir),
                    "method": method,
                },
                message="No fully validated last-known-good checkpoint is available for resume.",
                resolution=(
                    "Complete at least one stage boundary successfully and persist an atomic "
                    "last-known-good checkpoint before retrying resume."
                ),
            )
            raise RuntimeError(json.dumps(diagnostic))

        pointer = RecoveryCheckpointManager.load_last_known_good(output_dir)
        if pointer is not None:
            pointer_path = Path(pointer.get("checkpoint_path", "")).resolve()
            if pointer_path != selected.path.resolve():
                logger.warning(
                    "Last-known-good pointer does not match newest resumable checkpoint; "
                    "using newest validated checkpoint for strict auto-resume."
                )

        return selected
