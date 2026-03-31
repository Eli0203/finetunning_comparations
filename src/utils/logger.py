"""
Logger Utility
author: Eliana Vallejo
"""

import io
import logging
import sys
from typing import Protocol

class LoggerProtocol(Protocol):
    def info(self, msg: str, *args, **kwargs): ...
    def debug(self, msg: str, *args, **kwargs): ...
    def error(self, msg: str, *args, **kwargs): ...

class AppLogger:
    """UTF-8 safe application logger with console and file handlers."""

    def __init__(self, name: str = "FineTuningApp"):
        self._logger = logging.getLogger(name)
        if not self._logger.hasHandlers():
            self._logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Console: INFO and above — force UTF-8 so Unicode symbols (e.g. ✓)
            # do not crash on Windows where the default encoding is CP1252.
            if hasattr(sys.stdout, "buffer"):
                _stdout_utf8 = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                )
            else:
                _stdout_utf8 = sys.stdout  # already a non-binary stream (e.g. StringIO in tests)
            console = logging.StreamHandler(_stdout_utf8)
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self._logger.addHandler(console)
            
            # File: Full DEBUG history [1]
            file_h = logging.FileHandler('finetuning.log', mode='w')
            file_h.setLevel(logging.DEBUG)
            file_h.setFormatter(formatter)
            self._logger.addHandler(file_h)
            
        self._suppress_ml_noise()

    def _suppress_ml_noise(self):
        # Silencing noisy internal ML logs to maintain focus 
        for module in ["transformers", "peft", "datasets", "bitsandbytes"]:
            logging.getLogger(module).setLevel(logging.CRITICAL)

    def get_logger(self) -> logging.Logger:
        return self._logger

# Global Instance for Traceability 
logger = AppLogger().get_logger()