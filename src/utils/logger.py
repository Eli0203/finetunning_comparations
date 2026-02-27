"""
Logger Utility
author: Eliana Vallejo
"""

import logging
import sys
from typing import Protocol

class LoggerProtocol(Protocol):
    def info(self, msg: str, *args, **kwargs): ...
    def debug(self, msg: str, *args, **kwargs): ...
    def error(self, msg: str, *args, **kwargs): ...

class AppLogger:
    def __init__(self, name: str = "FineTuningApp"):
        self._logger = logging.getLogger(name)
        if not self._logger.hasHandlers():
            self._logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Console: INFO and above [1]
            console = logging.StreamHandler(sys.stdout)
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