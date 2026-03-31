import math
from typing import Any, Dict, Iterable, Tuple
import torch
from src.utils.math_utils import LaplaceMath
from src.utils.logger import logger

class LaplaceLoRAEngine:
    def __init__(self, lora_engine, prior_precision: float = 1.0, nkfac: int = 10):
        self.lora_engine = lora_engine
        self.prior_precision = prior_precision
        self.nkfac = nkfac
        # Almacenamos factores Kronecker por capa: {layer_name: (factor_A, factor_B)}
        self.curvature_factors = {}
        self._hook_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def accumulate_curvature(self, data_loader):
        """
        Implementa la acumulación incremental de curvatura.
        Calcula los bloques de Fisher para cada capa lineal adaptada [1, 8].
        """
        model = self._get_peft_model()
        model.eval()
        device = next(model.parameters()).device
        handles = self._register_factor_hooks(model)

        try:
            for batch in data_loader:
                model.zero_grad(set_to_none=True)
                self._hook_cache.clear()

                filtered_batch = self._prepare_batch(batch, device)
                outputs = model(**filtered_batch)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise ValueError("Laplace curvature accumulation requires model outputs with a loss attribute")

                loss.backward()

                for name, _module in self._iter_lora_a_modules(model):
                    a, g = self._get_factors_from_hooks(name)

                    if name not in self.curvature_factors:
                        self.curvature_factors[name] = (
                            torch.zeros((a.size(1), 0), device=device, dtype=a.dtype),
                            torch.zeros((g.size(1), 0), device=device, dtype=g.dtype),
                        )

                    # Actualización SVD incremental (Algoritmo 1) [2]
                    A_old, G_old = self.curvature_factors[name]
                    self.curvature_factors[name] = (
                        LaplaceMath.incremental_svd_update(A_old, a.t(), self.nkfac),
                        LaplaceMath.incremental_svd_update(G_old, g.t(), self.nkfac),
                    )
        finally:
            for handle in handles:
                handle.remove()

    def get_marginal_likelihood(self, log_likelihood: float) -> float:
        """
        Calcula la evidencia del modelo (Marginal Likelihood) para tunear λ [9, 10].
        P(y|X) ≈ exp(L(theta_MAP)) * (2pi)^(D/2) * |Σ|^1/2
        """
        total_log_det = 0.0
        model = self._get_peft_model()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for name, (A, G) in self.curvature_factors.items():
            # Sumamos los log-determinantes de cada bloque Kronecker [3]
            total_log_det += float(LaplaceMath.log_det_kfac_low_rank(
                A, G, 1.0/self.prior_precision, n_lora=self.lora_engine.lora_rank, d=A.size(0)
            ))

        # Ecuación 14: Log-Evidencia [5, 9]
        # Log P(y|X) = Log-Likelihood - 0.5 * λ * ||θ_MAP||^2 - 0.5 * LogDet(H) + const
        log_prior = -0.5 * self.prior_precision * self._get_weight_norm_sq()
        evidence = log_likelihood + log_prior - 0.5 * total_log_det + (n_params / 2) * math.log(2 * math.pi)
        return evidence

    def _get_peft_model(self):
        model = getattr(self.lora_engine, "peft_model", None)
        if model is None:
            raise ValueError("LaplaceLoRAEngine requires lora_engine.peft_model. Call apply_lora() first.")
        return model

    def _iter_lora_a_modules(self, model) -> Iterable[Tuple[str, torch.nn.Module]]:
        for name, module in model.named_modules():
            if "lora_A" in name and hasattr(module, "weight"):
                yield name, module

    def _register_factor_hooks(self, model):
        handles = []
        for name, module in self._iter_lora_a_modules(model):
            handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))
        if not handles:
            logger.warning("No LoRA A modules were found for Laplace curvature accumulation.")
        return handles

    def _make_forward_hook(self, name: str):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            activation = self._flatten_factor_tensor(inputs[0].detach())
            self._hook_cache.setdefault(name, {})["a"] = activation
        return hook

    def _make_backward_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if not grad_output:
                return
            gradient = self._flatten_factor_tensor(grad_output[0].detach())
            self._hook_cache.setdefault(name, {})["g"] = gradient
        return hook

    def _get_factors_from_hooks(self, name: str):
        cached = self._hook_cache.get(name, {})
        activation = cached.get("a")
        gradient = cached.get("g")
        if activation is None or gradient is None:
            raise RuntimeError(f"Missing Laplace hook factors for module '{name}'")
        return activation, gradient

    def _get_weight_norm_sq(self) -> float:
        model = self._get_peft_model()
        total = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total += float(torch.sum(param.detach() ** 2).item())
        return total

    def _prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        filtered_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            normalized_key = "labels" if key == "label" else key
            if normalized_key not in {"input_ids", "attention_mask", "token_type_ids", "labels"}:
                continue
            filtered_batch[normalized_key] = value.to(device) if torch.is_tensor(value) else value
        return filtered_batch

    @staticmethod
    def _flatten_factor_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor.reshape(-1, tensor.shape[-1])