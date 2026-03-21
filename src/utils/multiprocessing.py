"""Multiprocessing utilities.

This module contains helper primitives for inter-process communication
used throughout the finetuning pipeline.
"""

from dataclasses import dataclass
import multiprocessing as py_mp
import sys
import torch.multiprocessing as mp


@dataclass(frozen=True)
class MultiprocessingSetupResult:
    """Result object for startup multiprocessing configuration."""

    configured: bool
    requested_method: str
    current_method: str
    is_colab: bool
    message: str


class MultiprocessingContextConfigurator:
    """Configure process start method for consistent cross-platform behavior.

    Single Responsibility:
    - Ensure a single startup policy for multiprocessing context.
    - Report the resulting runtime state for diagnostics.
    """

    @staticmethod
    def configure_start_method(method: str = "spawn") -> MultiprocessingSetupResult:
        """Set process start method once at startup, with safe fallback reporting.

        Args:
            method: Start method to enforce globally. Defaults to ``spawn``.

        Returns:
            MultiprocessingSetupResult with runtime details.
        """
        is_colab = "google.colab" in sys.modules
        configured = False

        try:
            py_mp.set_start_method(method, force=True)
            configured = True
            message = f"Multiprocessing start method configured to '{method}'."
        except RuntimeError as exc:
            message = f"Could not set multiprocessing start method: {exc}"

        current = py_mp.get_start_method(allow_none=False)
        return MultiprocessingSetupResult(
            configured=configured,
            requested_method=method,
            current_method=current,
            is_colab=is_colab,
            message=message,
        )


def configure_spawn_context() -> MultiprocessingSetupResult:
    """Convenience helper for the project's required startup policy."""
    return MultiprocessingContextConfigurator.configure_start_method("spawn")


def get_spawn_context() -> mp.context.BaseContext:
    """Return the explicit spawn context used by IPC primitives."""
    return mp.get_context("spawn")


class DoubleBuffer:
    """A lightweight double-buffer for inter-process weight sharing.

    This buffer provides O(1) read access to the latest produced item.
    All IPC primitives are created with an explicit **spawn** context to prevent
    the ``SemLock`` fork/spawn mismatch that occurs when the system default
    context is ``fork`` (Linux) but worker processes are started with ``spawn``.
    """

    def __init__(self):
        # Explicitly use spawn context so that Manager and Lock are spawn-safe
        # regardless of the process-global start method default.
        ctx = get_spawn_context()
        manager = ctx.Manager()
        self._slots = manager.list([None, None])
        self._current = manager.Value("i", 0)
        self._lock = ctx.Lock()

    def put(self, item):
        """Store an item in the next slot and update the latest pointer."""
        with self._lock:
            next_idx = 1 - self._current.value
            self._slots[next_idx] = item
            self._current.value = next_idx

    def get_latest(self):
        """Return the most recently written item (or None)."""
        with self._lock:
            return self._slots[self._current.value]
