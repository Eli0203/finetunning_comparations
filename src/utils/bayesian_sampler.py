"""Bayesian causal sampler scaffolding for LoRA weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.hypernetwork import LoRAHypernetwork
from src.utils.logger import logger


@dataclass
class MixtureComponents:
    """Container for MoG parameters per causal path."""

    pi: torch.Tensor
    mu: torch.Tensor
    precision: torch.Tensor


class BayesianCausalSampler:
    """Duck-type compatible Bayesian sampler shell for causal training."""

    def __init__(
        self,
        causal_engine: Any,
        model: nn.Module,
        device: str = "cpu",
        n_components: int = 4,
        enable_pg_pos: bool = False,
        kfac_correlation: bool = False,
        random_dirichlet_init: bool = True,
    ) -> None:
        if n_components < 2:
            raise ValueError(f"n_components must be >= 2, got {n_components}")

        self.causal_engine: Optional[Any] = causal_engine
        self.device = device
        self.n_components = n_components
        self.enable_pg_pos = enable_pg_pos
        self.kfac_correlation = kfac_correlation
        self.random_dirichlet_init = random_dirichlet_init

        self._param_specs: Dict[str, Tuple[torch.Size, torch.dtype]] = {
            name: (param.shape, param.dtype)
            for name, param in model.named_parameters()
        }
        self._lora_param_specs: Dict[str, Tuple[torch.Size, torch.dtype]] = {
            name: spec
            for name, spec in self._param_specs.items()
            if "lora" in name.lower()
        }

        self.path_weights = self._compute_path_weights()
        self._mixture_components: Dict[str, MixtureComponents] = {}
        self._kfac_blocks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        lora_rank = max(1, int(getattr(getattr(causal_engine, "lora_engine", None), "lora_rank", 8)))
        self._wishart_rank = lora_rank
        self._scale_V = nn.ParameterDict(
            {
                key: nn.Parameter(torch.ones(lora_rank, dtype=torch.float32) / float(lora_rank))
                for key in self.path_weights
            }
        )

        self._hypernetwork: Optional[LoRAHypernetwork] = None
        if self.enable_pg_pos:
            self._hypernetwork = LoRAHypernetwork(
                n_layers=max(1, len(self._lora_param_specs)),
                out_dim=lora_rank,
            )

        logger.info(
            "BayesianCausalSampler initialized | components=%s pg_pos=%s kfac=%s",
            self.n_components,
            self.enable_pg_pos,
            self.kfac_correlation,
        )

    def _compute_path_weights(self) -> Dict[str, float]:
        allocation = getattr(self.causal_engine, "budget_allocation", None)
        if not allocation:
            return {"default": 1.0}

        total = float(sum(allocation.values()))
        if total <= 0:
            n = len(allocation)
            return {path: 1.0 / n for path in allocation}

        return {path: float(count) / total for path, count in allocation.items()}

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("causal_engine", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.causal_engine = None

    def refresh_path_weights(self) -> None:
        """Refresh normalized path weights from the causal engine allocation."""
        if self.causal_engine is None:
            logger.warning("BayesianCausalSampler refresh skipped: causal_engine unavailable")
            return
        self.path_weights = self._compute_path_weights()

    def _ensure_scale_param(self, path_key: str) -> nn.Parameter:
        """Ensure path-specific diagonal Wishart scale parameter exists."""
        if path_key not in self._scale_V:
            self._scale_V[path_key] = nn.Parameter(
                torch.ones(self._wishart_rank, dtype=torch.float32) / float(self._wishart_rank)
            )
        return self._scale_V[path_key]

    def _sample_precision_matrices(self, path_key: str) -> torch.Tensor:
        """Sample precision matrices from a Wishart prior per component."""
        rank = self._wishart_rank
        raw_diag = self._ensure_scale_param(path_key)
        diag_scale = torch.clamp(raw_diag.detach().cpu().to(torch.float32), min=1e-4)
        scale_tril = torch.diag(diag_scale)
        dof = float(rank + 1)

        try:
            wishart = torch.distributions.Wishart(df=dof, scale_tril=scale_tril)
            precision = wishart.sample((self.n_components,)).to(torch.float32)
            if precision.ndim != 3:
                raise ValueError("Wishart returned unexpected precision shape")
            return precision
        except Exception as exc:
            logger.warning(
                "Wishart sampling failed for path '%s' (%s). Falling back to diagonal precision.",
                path_key,
                exc,
            )
            eye = torch.eye(rank, dtype=torch.float32)
            return eye.unsqueeze(0).repeat(self.n_components, 1, 1)

    def update_kfac_blocks(
        self,
        name: str,
        A_factor: torch.Tensor,
        G_factor: torch.Tensor,
    ) -> None:
        """Store KFAC blocks for later correlation-aware sampling."""
        self._kfac_blocks[name] = (
            A_factor.detach().cpu().to(torch.float32),
            G_factor.detach().cpu().to(torch.float32),
        )

    def _select_path_key(self, param_name: str) -> str:
        for key in self.path_weights:
            if key != "default" and key in param_name:
                return key
        return "default"

    def _ensure_components(self, path_key: str, numel: int) -> MixtureComponents:
        if path_key in self._mixture_components:
            return self._mixture_components[path_key]

        if self.random_dirichlet_init:
            alpha = torch.rand(self.n_components, dtype=torch.float32).clamp_min(1e-3)
            pi = torch.distributions.Dirichlet(alpha).sample()
        else:
            pi = torch.ones(self.n_components, dtype=torch.float32) / float(self.n_components)

        mu = torch.zeros(self.n_components, numel, dtype=torch.float32)
        if self._hypernetwork is not None:
            base = self._hypernetwork.generate(0, 0).detach().cpu().to(torch.float32)
            if base.numel() > 0:
                repeats = (numel + base.numel() - 1) // base.numel()
                tiled = base.repeat(repeats)[:numel]
                mu = tiled.repeat(self.n_components, 1)

        precision = self._sample_precision_matrices(path_key)

        components = MixtureComponents(pi=pi, mu=mu, precision=precision)
        self._mixture_components[path_key] = components
        return components

    def sample_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Sample one batch of CPU weight tensors in a MoG-compatible scaffold."""
        del batch_size
        state_dict: Dict[str, torch.Tensor] = {}
        specs = self._lora_param_specs if self._lora_param_specs else self._param_specs

        for name, (shape, dtype) in specs.items():
            numel = int(torch.tensor(shape).prod().item())
            path_key = self._select_path_key(name)
            components = self._ensure_components(path_key, numel)

            comp_idx = int(torch.multinomial(components.pi, num_samples=1).item())
            base_mean = components.mu[comp_idx]
            scale = float(self.path_weights.get(path_key, 1.0)) ** 0.5
            sampled_flat = base_mean + torch.randn(numel, dtype=torch.float32) * scale

            sampled = sampled_flat.reshape(shape).to(dtype=dtype).cpu()
            if not torch.isfinite(sampled).all():
                logger.warning("Non-finite sample detected for %s; replacing with zeros", name)
                sampled = torch.zeros(shape, dtype=dtype)
            state_dict[name] = sampled

        return state_dict

    def get_config(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "n_components": self.n_components,
            "enable_pg_pos": self.enable_pg_pos,
            "kfac_correlation": self.kfac_correlation,
            "random_dirichlet_init": self.random_dirichlet_init,
            "num_paths": len(self.path_weights),
            "num_kfac_blocks": len(self._kfac_blocks),
        }
