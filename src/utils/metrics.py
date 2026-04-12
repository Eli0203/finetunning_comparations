import torch
import torch.nn.functional as F
import evaluate
from typing import Dict, Optional
import json
from datetime import datetime, timezone


METRIC_SCHEMA = (
    "accuracy",
    "f1",
    "nll",
    "ece",
    "posterior_mean_nll",
    "posterior_variance_mean",
    "epistemic_uncertainty_mean",
)

SUPPORTED_METHODS = {
    "lora",
    "laplace_lora",
    "causal_lora",
    "laplace",
    "causal",
}

LAPLACE_METHODS = {"laplace_lora", "laplace"}

class UnifiedEvaluator:
    """Standardized evaluation engine for LoRA and Bayesian Laplace-LoRA [3]."""
    def __init__(self, task_name: str):
        """Initialize GLUE metric adapters and calibration settings for one task."""
        self.task_name = task_name.lower()
        self.glue_metric = evaluate.load("glue", self.task_name) # [11]
        self.n_bins = 10 # Standard for ECE computation [5]

    @staticmethod
    def _build_diagnostic(
        error_type: str,
        message: str,
        context: Optional[Dict[str, object]] = None,
    ) -> str:
        payload = {
            "error_type": error_type,
            "level": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or {},
            "message": message,
        }
        return json.dumps(payload)

    @staticmethod
    def _normalize_method(method: str) -> str:
        normalized = method.strip().lower()
        if normalized not in SUPPORTED_METHODS:
            raise ValueError(
                UnifiedEvaluator._build_diagnostic(
                    error_type="unsupported_method",
                    message=f"Unsupported method '{method}'.",
                    context={"supported_methods": sorted(SUPPORTED_METHODS)},
                )
            )
        return normalized

    @staticmethod
    def _validate_inputs(logits: torch.Tensor, labels: torch.Tensor) -> None:
        if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(
                UnifiedEvaluator._build_diagnostic(
                    error_type="invalid_type",
                    message="logits and labels must be torch.Tensor instances.",
                    context={
                        "logits_type": str(type(logits)),
                        "labels_type": str(type(labels)),
                    },
                )
            )

        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                UnifiedEvaluator._build_diagnostic(
                    error_type="invalid_shape",
                    message="Expected logits shape [N, C] and labels shape [N].",
                    context={
                        "logits_shape": list(logits.shape),
                        "labels_shape": list(labels.shape),
                    },
                )
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                UnifiedEvaluator._build_diagnostic(
                    error_type="shape_mismatch",
                    message="Batch dimension mismatch between logits and labels.",
                    context={
                        "logits_batch": int(logits.shape[0]),
                        "labels_batch": int(labels.shape[0]),
                    },
                )
            )

        if labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(
                UnifiedEvaluator._build_diagnostic(
                    error_type="invalid_label_dtype",
                    message="labels tensor must contain integer class indices.",
                    context={"labels_dtype": str(labels.dtype)},
                )
            )

    @staticmethod
    def _macro_f1(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        classes = torch.unique(torch.cat([predictions, labels])).tolist()
        f1_scores = []
        for cls in classes:
            cls_t = torch.tensor(cls, device=labels.device)
            tp = ((predictions == cls_t) & (labels == cls_t)).sum().item()
            fp = ((predictions == cls_t) & (labels != cls_t)).sum().item()
            fn = ((predictions != cls_t) & (labels == cls_t)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append((2 * precision * recall) / (precision + recall))

        return float(sum(f1_scores) / max(len(f1_scores), 1))

    def compute_all(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        method: str = "lora",
    ) -> Dict[str, Optional[float]]:
        normalized_method = self._normalize_method(method)
        self._validate_inputs(logits, labels)

        predictions = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        confidences, _ = torch.max(probs, dim=1)

        accuracy = float((predictions == labels).float().mean().item())
        f1 = self._macro_f1(predictions, labels)
        nll = float(F.cross_entropy(logits, labels).item())
        ece = self._calculate_ece(confidences, predictions, labels)

        # Keep existing GLUE integration for additional task-specific keys.
        glue_results = self.glue_metric.compute(
            predictions=predictions.cpu().numpy(),
            references=labels.cpu().numpy(),
        )

        results: Dict[str, Optional[float]] = {
            "accuracy": accuracy,
            "f1": f1,
            "nll": nll,
            "ece": ece,
            "posterior_mean_nll": None,
            "posterior_variance_mean": None,
            "epistemic_uncertainty_mean": None,
        }

        # Preserve any extra GLUE metrics (e.g. matthews_correlation, pearson).
        for key, value in glue_results.items():
            if key not in {"accuracy", "f1"}:
                results[key] = float(value) if isinstance(value, (int, float)) else value

        if normalized_method in LAPLACE_METHODS:
            entropy = -(
                probs * probs.clamp(min=1e-12).log()
            ).sum(dim=-1).mean()
            results["posterior_mean_nll"] = nll
            results["posterior_variance_mean"] = float(probs.var(dim=-1).mean().item())
            results["epistemic_uncertainty_mean"] = float(entropy.item())

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


def natural_indirect_effect(
    p_y_given_x_m: torch.Tensor,
    p_m_x: torch.Tensor,
    p_m_x_prime: torch.Tensor,
    m_dim: int = -1,
) -> torch.Tensor:
    """Compute the Natural Indirect Effect (NIE) via mediation analysis.

    NIE measures the effect of X on Y that is mediated through M (the mediator).

    Formula (based on Pearl's mediation decomposition):
        NIE = E[Y_{x, M_{x'}}] - E[Y_{x, M_x}]

    Args:
        p_y_given_x_m: Tensor representing P(Y | X=x, M=m).
        p_m_x: Tensor representing P(M=m | X=x).
        p_m_x_prime: Tensor representing P(M=m | X=x').
        m_dim: Dimension index corresponding to the mediator M.

    Returns:
        Tensor containing the NIE for each X/evaluation point.
    """

    # Ensure the mediator distributions are normalized
    p_m_x = p_m_x / p_m_x.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p_m_x_prime = p_m_x_prime / p_m_x_prime.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    # Align shapes for broadcasting
    target_shape = [1] * p_y_given_x_m.ndim
    target_shape[m_dim] = -1
    p_m_x_expanded = p_m_x.view(*target_shape)
    p_m_x_prime_expanded = p_m_x_prime.view(*target_shape)

    e_y_x_mprime = (p_y_given_x_m * p_m_x_prime_expanded).sum(dim=m_dim)
    e_y_x_m = (p_y_given_x_m * p_m_x_expanded).sum(dim=m_dim)

    return e_y_x_mprime - e_y_x_m
