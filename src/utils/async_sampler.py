"""Asynchronous sampling utilities.

This module provides a background sampler for decoupling weight generation
from the main training loop using multiprocessing.

Supports both random sampling (legacy) and causal-aware sampling (new).
"""

import traceback

import torch
import torch.multiprocessing as mp
from typing import Any, Dict, Iterator, Optional, Tuple
from src.utils.logger import logger


def _random_weight_generator(
    param_specs: Dict[str, Tuple[torch.Size, torch.dtype]],
    max_steps: int = 100,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Generate a stream of random CPU weight tensors from parameter specs.

    Accepts a metadata dict (name → (shape, dtype)) instead of the live model
    so the generator is safe to call inside a spawned worker process.
    """
    for _ in range(max_steps):
        yield {
            name: torch.randn(shape, dtype=dtype)
            for name, (shape, dtype) in param_specs.items()
        }


class BackgroundSampler:
    """Asynchronously generates weight updates using multiprocessing.

    Single Responsibility:
    - Manage subprocess lifecycle
    - Call sampler repeatedly and push results to the shared buffer
    - Does NOT implement weight generation logic

    Spawn-safe design:
    - ``self.model`` is **not** stored; only per-parameter metadata
      (shapes, dtypes) is retained in ``_param_specs``.  This prevents
      CUDA tensors from being pickled when the worker is spawned.
    - ``CausalWeightSampler`` must also be spawn-safe (it implements its
      own ``__getstate__``/``__setstate__`` to strip CUDA refs).
    """

    def __init__(
        self,
        buffer,
        model: torch.nn.Module,
        max_steps: int = 100,
        causal_sampler=None,  # Optional: CausalWeightSampler instance
    ) -> None:
        """
        Initialize background sampler.

        Args:
            buffer: DoubleBuffer instance for storing sampled weights.
            model: PyTorch model to sample weights for.  Only metadata
                   (parameter shapes and dtypes) is retained; the live model
                   is not stored to keep this object spawn-safe.
            max_steps: Maximum number of sampling iterations.
            causal_sampler: Optional CausalWeightSampler for causal-aware
                            sampling.  If None, falls back to random CPU tensors.
        """
        self.buffer = buffer
        # Extract CPU-safe parameter metadata; do NOT hold the live model
        # (which may be on CUDA and cannot be pickled for spawn workers).
        self._param_specs: Dict[str, Tuple[torch.Size, torch.dtype]] = {
            name: (param.shape, param.dtype)
            for name, param in model.named_parameters()
        }
        self._context = mp.get_context("spawn")
        self._shared_state_manager = self._context.Manager()
        self._error_state = self._shared_state_manager.dict(last_error=None)
        self._stop_event = self._context.Event()
        self.max_steps = max_steps
        self.causal_sampler = causal_sampler  # Dependency injection
        self.process: Optional[mp.Process] = None
        self._last_error: Optional[str] = None
        self._stop_requested = False

        if causal_sampler is not None:
            logger.info("BackgroundSampler initialized with causal-aware sampler")
        else:
            logger.info("BackgroundSampler initialized with random sampler (legacy)")

    def __getstate__(self) -> Dict[str, Any]:
        """Return worker-safe state for spawn pickling.

        The child worker does not need the parent-only process management objects
        such as the spawn context, sync manager, or current Process handle.
        """
        state = self.__dict__.copy()
        state.pop('_context', None)
        state.pop('_shared_state_manager', None)
        state.pop('process', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore worker state after spawn deserialization."""
        self.__dict__.update(state)
        self._context = mp.get_context("spawn")
        self._shared_state_manager = None
        self.process = None

    def _worker(self, max_steps: int) -> None:
        """Worker process that samples and buffers weights.
        
        Uses either causal or random sampling based on whether sampler is available.
        """
        try:
            if self.causal_sampler is not None:
                self._causal_sampling_worker(max_steps)
            else:
                self._random_sampling_worker(max_steps)
        except Exception:
            error_message = traceback.format_exc()
            try:
                self._error_state['last_error'] = error_message
            except Exception:
                logger.error("Failed to publish worker error to parent process")
            raise
    
    def _causal_sampling_worker(self, max_steps: int) -> None:
        """Worker using causal-aware sampling."""
        for step in range(max_steps):
            if self._stop_event.is_set():
                return
            try:
                weights = self.causal_sampler.sample_batch()
                self.buffer.put(weights)
                if step % 10 == 0:
                    logger.debug(f"Causal sampling: step {step}/{max_steps}")
            except Exception as e:
                raise RuntimeError(
                    f"Error in causal sampling worker at step {step}: {e}"
                ) from e
    
    def _random_sampling_worker(self, max_steps: int) -> None:
        """Worker using random sampling (legacy). Always generates CPU tensors."""
        for weights in _random_weight_generator(self._param_specs, max_steps=max_steps):
            if self._stop_event.is_set():
                return
            self.buffer.put(weights)

    def _refresh_last_error(self) -> None:
        """Refresh cached worker error from shared multiprocessing state."""
        shared_error = self._error_state.get('last_error')
        if shared_error is not None:
            self._last_error = shared_error

    def get_last_error(self) -> Optional[str]:
        """Return the most recent worker traceback captured from the child process."""
        self._refresh_last_error()
        return self._last_error

    def get_status(self) -> Dict[str, Any]:
        """Return process health details for diagnostics and error handling."""
        self._refresh_last_error()
        return {
            'is_running': bool(self.process and self.process.is_alive()),
            'exitcode': self.process.exitcode if self.process is not None else None,
            'last_error': self._last_error,
            'stop_requested': self._stop_requested,
        }

    def raise_if_failed(self) -> None:
        """Raise a RuntimeError if the worker process has reported a failure."""
        self._refresh_last_error()
        if self._last_error is not None:
            raise RuntimeError(f"Background sampler worker failed:\n{self._last_error}")

        if (
            self.process is not None
            and not self._stop_requested
            and not self.process.is_alive()
            and self.process.exitcode not in (None, 0)
        ):
            raise RuntimeError(
                "Background sampler process exited unexpectedly "
                f"with exit code {self.process.exitcode}."
            )

    def start(self) -> None:
        """Start the background sampling process."""
        self._stop_requested = False
        self._stop_event.clear()
        self.process = self._context.Process(target=self._worker, args=(self.max_steps,))
        self.process.daemon = True
        self.process.start()
        logger.info("Background sampler process started")
        self.process.join(timeout=0.05)
        self.raise_if_failed()

    def stop(self) -> None:
        """Stop the background sampling process."""
        if self.process is not None:
            self._stop_requested = True
            self._stop_event.set()
            if self.process.pid is not None or self.process.exitcode is not None:
                self.process.join(timeout=1)
            if self.process.pid is not None and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            logger.info("Background sampler process stopped")
