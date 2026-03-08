from typing import Dict, Optional
import torch
from src.utils.math_utils import LaplaceMath
from src.utils.logger import logger
# CORRECTED: Now used for Dependency Injection and metadata access
from src.finetuner.lora_engine import FineTuningEngine

class LaplaceLoRAEngine:
    def __init__(self, lora_engine, prior_precision: float = 1.0, nkfac: int = 10):
        self.lora_engine = lora_engine
        self.prior_precision = prior_precision
        self.nkfac = nkfac
        # Almacenamos factores Kronecker por capa: {layer_name: (factor_A, factor_B)}
        self.curvature_factors = {} 

    def accumulate_curvature(self, data_loader):
        """
        Implementa la acumulación incremental de curvatura.
        Calcula los bloques de Fisher para cada capa lineal adaptada [1, 8].
        """
        self.lora_engine.peft_model.eval()
        device = next(self.lora_engine.peft_model.parameters()).device
        
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            # 1. Forward pass para obtener actividades (a_l-1)
            # 2. Backward pass para obtener gradientes w.r.t outputs (g_l) [8]
            
            for name, module in self.lora_engine.peft_model.named_modules():
                if "lora_A" in name: # Capturamos factores para adaptadores
                    # Supongamos que capturamos actividades 'a' y gradientes 'g' vía hooks
                    a, g = self._get_factors_from_hooks(name) 
                    
                    if name not in self.curvature_factors:
                        self.curvature_factors[name] = (
                            torch.zeros(a.size(1), self.nkfac).to(device),
                            torch.zeros(g.size(1), self.nkfac).to(device)
                        )
                    
                    # Actualización SVD incremental (Algoritmo 1) [2]
                    A_old, G_old = self.curvature_factors[name]
                    self.curvature_factors[name] = (
                        LaplaceMath.incremental_svd_update(A_old, a.t(), self.nkfac),
                        LaplaceMath.incremental_svd_update(G_old, g.t(), self.nkfac)
                    )

    def get_marginal_likelihood(self, log_likelihood: float) -> float:
        """
        Calcula la evidencia del modelo (Marginal Likelihood) para tunear λ [9, 10].
        P(y|X) ≈ exp(L(theta_MAP)) * (2pi)^(D/2) * |Σ|^1/2
        """
        total_log_det = 0.0
        n_params = sum(p.numel() for p in self.lora_engine.peft_model.parameters() if p.requires_grad)
        
        for name, (A, G) in self.curvature_factors.items():
            # Sumamos los log-determinantes de cada bloque Kronecker [3]
            total_log_det += LaplaceMath.log_det_kfac_low_rank(
                A, G, 1.0/self.prior_precision, n_lora=self.lora_engine._config.r, d=A.size(0)
            )
            
        # Ecuación 14: Log-Evidencia [5, 9]
        # Log P(y|X) = Log-Likelihood - 0.5 * λ * ||θ_MAP||^2 - 0.5 * LogDet(H) + const
        log_prior = -0.5 * self.prior_precision * self._get_weight_norm_sq()
        evidence = log_likelihood + log_prior - 0.5 * total_log_det + (n_params / 2) * math.log(2 * math.pi)
        return evidence