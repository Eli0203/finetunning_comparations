import torch
import torch.nn.functional as F
import evaluate
import numpy as np
from typing import Dict, Any
from src.utils.logger import logger

class UnifiedEvaluator:
    """Standardized evaluation engine for LoRA and Bayesian Laplace-LoRA [3]."""
    def __init__(self, task_name: str):
        self.task_name = task_name.lower()
        self.glue_metric = evaluate.load("glue", self.task_name) # [11]
        self.n_bins = 10 # Standard for ECE computation [5]

    def compute_all(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        # 1. GLUE Task Metrics
        if self.task_name == "sts-b":
            preds = logits.squeeze().cpu().numpy() # Regression
        else:
            preds = torch.argmax(logits, dim=-1).cpu().numpy() # Classification [12]
        
        refs = labels.cpu().numpy()
        results = self.glue_metric.compute(predictions=preds, references=refs) # [13]

        # 2. Uncertainty Metrics (Bayesian Focus)
        # These reveal the calibration benefits of Laplace-LoRA [6, 7]
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # NLL: Negative Log-Likelihood [5]
        nll = F.cross_entropy(logits, labels).item()
        
        # ECE: Expected Calibration Error [5, 14]
        ece = self._calculate_ece(confidences, predictions, labels)
        
        results.update({"nll": nll, "ece": ece})
        return results

    def _calculate_ece(self, confidences, predictions, labels) -> float:
        """Computes ECE by binning predictions by confidence [5]."""
        ece = torch.zeros(1, device=confidences.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        
        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()