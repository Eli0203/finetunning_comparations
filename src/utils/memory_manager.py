"""
Memory management Utility
author: Eliana Vallejo
"""


import psutil
import torch
import gc
from typing import Any
from src.utils.logger import logger
from src.utils.multiprocessing import RingBuffer


class MemoryOptimizer:
    """Utility for memory management and traceability.
    Incorporates PyTorch best practices for memory-efficient fine-tuning
    """

    @staticmethod
    def log_resource_usage(phase: str):
        """Monitor RAM and VRAM usage to measure computational complexity"""
        process = psutil.Process()
        ram_gb = process.memory_info().rss / (1024**3)
        logger.info(f"[{phase}] Current RAM Usage: {ram_gb:.2f} GB")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"[{phase}] Current VRAM Usage: {vram_gb:.2f} GB")

    @staticmethod
    def cleanup():
        """Clear Python and CUDA caches to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup completed: GC and Cache cleared.")

    @staticmethod
    def create_double_buffer():
        """Create a cross-process ring buffer for async sampling.

        Returns:
            RingBuffer: Provides O(1) retrieval of the latest item.
        """
        return RingBuffer(size=15)

    @staticmethod
    def create_ring_buffer(size: int = 15) -> RingBuffer:
        """Create a configurable ring buffer for async sampling."""
        return RingBuffer(size=size)

    @staticmethod
    def try_get_latest(buffer: Any):
        """Attempt to read the latest item from a DoubleBuffer/RingBuffer."""
        return buffer.get_latest()
