
"""
Utility for computing GLUE task metrics using the 'evaluate' library
Author: Eliana Vallejo
"""
from typing import Dict, Any, List, Protocol
import torch
import evaluate
import numpy as np

from src.utils.logger import logger

class MetricCalculator(Protocol):
    """Protocol for metric computation to ensure loose coupling."""
    def compute(self, predictions: torch.Tensor, references: torch.Tensor) -> Dict[str, float]:
        """Computes the metric given model predictions and true labels."""
        return None
       

class GLUEMetrics:
    """
    Utility to handle specific GLUE task metrics.
    Follows the strategy of using the 'evaluate' library for standardization.
    """
    def __init__(self, task_name: str):
        self.task_name = task_name.lower()
        # The 'evaluate' library handles the specific GLUE task requirements [5, 7]
        self.metric = evaluate.load("glue", self.task_name)

    def _preprocess(self, logits: torch.Tensor) -> np.ndarray:
        """Converts model logits to discrete predictions."""
        # For classification tasks, we take the index of the highest probability [8, 9]
        if self.task_name != "sts-b": # STS-B is a regression task [10, 11]
            return torch.argmax(logits, dim=-1).cpu().numpy()
        return logits.squeeze().cpu().numpy()

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Computes the task-specific metrics (e.g., Accuracy/F1 for MRPC).
        """
        logger.info(f"Computing metrics for task: {self.task_name}") # Traceability
        preds = self._preprocess(logits)
        refs = labels.cpu().numpy()
        
        # Returns metrics like {'accuracy': 0.95, 'f1': 0.94} 
        results = self.metric.compute(predictions=preds, references=refs)
        logger.info(f"Final Metrics: {results}") # Traceability
        return results

def compute_glue_metrics(task_name: str, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Functional wrapper for the GLUEMetrics strategy.
    Facilitates testing and standardizes evaluation across the benchmark.
    """
    calculator = GLUEMetrics(task_name)
    return calculator.compute(logits, labels)