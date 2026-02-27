import psutil
import torch
import gc
from src.utils.logger import logger

class MemoryOptimizer:
    """
    Utility for memory management and traceability.
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
        """
        Clears Python's object-specific allocators and the PyTorch VRAM cache.
        Essential for returning memory arenas to the Operating System [4, 8].
        """
        # Force the collector to release unreferenced objects
        gc.collect()
        if torch.cuda.is_available():
            # Release unused cached memory currently held by the GPU allocator
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup completed: GC and Cache cleared.")