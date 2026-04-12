"""Settings package with SRP-separated modules.

Public API exports all configuration classes and the factory, with each module
responsible for a single concern:
- base.py: Shared Settings base class
- experiment_settings.py: Experiment-specific subclasses (LoRA, Laplace-LoRA, Causal-LoRA)
- factory.py: SettingsFactory for creating fresh instances
- causal_config.py: CausalTrainingConfig for causal orchestration
"""

from .base import Settings
from .experiment_settings import (
    ExperimentType,
    BaseRunSettings,
    LoraSettings,
    LaplaceLoraSettings,
    CausalLoraSettings,
)
from .factory import SettingsFactory
from .causal_config import CausalTrainingConfig

__all__ = [
    "Settings",
    "ExperimentType",
    "BaseRunSettings",
    "LoraSettings",
    "LaplaceLoraSettings",
    "CausalLoraSettings",
    "SettingsFactory",
    "CausalTrainingConfig",
]
