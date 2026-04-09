"""Utility modules for causal LoRA fine-tuning."""

from src.utils.causal_sampler import CausalWeightSampler
from src.utils.async_sampler import BackgroundSampler
from src.utils.training_integrator import ContinuousWeightApplier, TrainingBudgetMonitor
from src.utils.logger import logger
from src.utils.math_utils import CausalMath, LaplaceMath
from src.utils.metrics import natural_indirect_effect
from src.utils.memory_manager import MemoryOptimizer
from src.utils.multiprocessing import DoubleBuffer

try:
    from src.utils.multiprocessing import RingBuffer
except ImportError:
    RingBuffer = None

try:
    from src.utils.bayesian_sampler import BayesianCausalSampler
except ImportError:
    BayesianCausalSampler = None

try:
    from src.utils.hypernetwork import LoRAHypernetwork
except ImportError:
    LoRAHypernetwork = None

__all__ = [
    'CausalWeightSampler',
    'BackgroundSampler',
    'ContinuousWeightApplier',
    'TrainingBudgetMonitor',
    'logger',
    'CausalMath',
    'LaplaceMath',
    'natural_indirect_effect',
    'MemoryOptimizer',
    'DoubleBuffer',
    'RingBuffer',
    'BayesianCausalSampler',
    'LoRAHypernetwork',
]
