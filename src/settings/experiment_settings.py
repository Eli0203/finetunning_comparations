"""Experiment-specific settings models with discriminator-based dispatch.

Provides BaseRunSettings as a shared foundation, plus three experiment-specific
subtypes (LoraSettings, LaplaceLoraSettings, CausalLoraSettings) that enforce
algorithm-specific constraints and canonical artifact paths.
"""

from pydantic import Field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .base import Settings


ExperimentType = Literal["lora", "laplace_lora", "causal_lora"]


class BaseRunSettings(Settings):
    """Shared run settings with experiment discriminator and canonical artifact paths.
    
    Enforces single responsibility: common fields and properties for all
    experiment types, with discriminator-driven dispatch to subtypes.
    """

    experiment_type: ExperimentType = Field(default="lora", validation_alias="EXPERIMENT_TYPE")
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12], validation_alias="RUN_ID")

    @property
    def canonical_artifact_root(self) -> Path:
        """Canonical artifact root for checkpoints, metrics, and logs.
        
        Format: output/{experiment_type}/{task_name}/{run_id}/
        """
        return Path("output") / self.experiment_type / self.task_name / self.run_id

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for experiment-specific checkpoints."""
        return self.canonical_artifact_root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        """Directory for experiment-specific metrics outputs."""
        return self.canonical_artifact_root / "metrics"

    @property
    def logs_dir(self) -> Path:
        """Directory for experiment-specific training logs."""
        return self.canonical_artifact_root / "logs"


class LoraSettings(BaseRunSettings):
    """LoRA-specific settings with experiment_type binding.
    
    Single responsibility: LoRA-only configuration validation and defaults.
    """
    experiment_type: Literal["lora"] = "lora"


class LaplaceLoraSettings(BaseRunSettings):
    """Laplace-LoRA-specific settings with experiment_type binding.
    
    Single responsibility: Laplace-specific configuration validation and defaults.
    """
    experiment_type: Literal["laplace_lora"] = "laplace_lora"


class CausalLoraSettings(BaseRunSettings):
    """Causal-LoRA-specific settings with experiment_type binding.
    
    Single responsibility: Causal-specific configuration validation and defaults.
    """
    experiment_type: Literal["causal_lora"] = "causal_lora"
