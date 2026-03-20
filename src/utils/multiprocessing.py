"""Multiprocessing utilities.

This module contains helper primitives for inter-process communication
used throughout the finetuning pipeline.
"""

import torch.multiprocessing as mp


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
        ctx = mp.get_context("spawn")
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
