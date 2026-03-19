"""Asynchronous sampling utilities.

This module provides a background sampler for decoupling weight generation
from the main training loop using multiprocessing.

Supports both random sampling (legacy) and causal-aware sampling (new).
"""

import torch
import torch.multiprocessing as mp
from typing import Dict, Iterator, Optional
from src.utils.logger import logger


def _random_weight_generator(
    model: torch.nn.Module,
    max_steps: int = 100,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Generate a stream of random weight tensors for async sampling."""
    for _ in range(max_steps):
        yield {k: torch.randn_like(v) for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}


class BackgroundSampler:
    """Asynchronously generates weight updates using multiprocessing.
    
    Single Responsibility:
    - Manage subprocess lifecycle
    - Call sampler repeatedly
    - Push results to buffer
    - Does NOT implement weight generation logic
    
    Supports two sampling modes:
    1. Random (legacy): Uses _random_weight_generator
    2. Causal-aware (new): Uses CausalWeightSampler for informed sampling
    """

    def __init__(
        self,
        buffer,
        model: torch.nn.Module,
        max_steps: int = 100,
        causal_sampler = None  # Optional: CausalWeightSampler instance
    ):
        """
        Initialize background sampler.
        
        Args:
            buffer: DoubleBuffer instance for storing sampled weights
            model: PyTorch model to sample weights for
            max_steps: Maximum number of sampling iterations
            causal_sampler: Optional CausalWeightSampler for causal-aware sampling.
                          If None, falls back to random sampling.
        """
        self.buffer = buffer
        self.model = model
        self.max_steps = max_steps
        self.causal_sampler = causal_sampler  # Dependency injection
        self.process: Optional[mp.Process] = None
        
        if causal_sampler is not None:
            logger.info("BackgroundSampler initialized with causal-aware sampler")
        else:
            logger.info("BackgroundSampler initialized with random sampler (legacy)")

    def _worker(self, max_steps: int) -> None:
        """Worker process that samples and buffers weights.
        
        Uses either causal or random sampling based on whether sampler is available.
        """
        if self.causal_sampler is not None:
            # Causal-aware sampling: use provided sampler
            self._causal_sampling_worker(max_steps)
        else:
            # Legacy random sampling
            self._random_sampling_worker(max_steps)
    
    def _causal_sampling_worker(self, max_steps: int) -> None:
        """Worker using causal-aware sampling."""
        for step in range(max_steps):
            try:
                weights = self.causal_sampler.sample_batch()
                self.buffer.put(weights)
                if step % 10 == 0:
                    logger.debug(f"Causal sampling: step {step}/{max_steps}")
            except Exception as e:
                logger.error(f"Error in causal sampling worker at step {step}: {e}")
                break
    
    def _random_sampling_worker(self, max_steps: int) -> None:
        """Worker using random sampling (legacy)."""
        for weights in _random_weight_generator(self.model, max_steps=max_steps):
            self.buffer.put(weights)

    def start(self) -> None:
        """Start the background sampling process."""
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.process = mp.Process(target=self._worker, args=(self.max_steps,))
        self.process.daemon = True
        self.process.start()
        logger.info("Background sampler process started")

    def stop(self) -> None:
        """Stop the background sampling process."""
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=1)
            logger.info("Background sampler process stopped")
