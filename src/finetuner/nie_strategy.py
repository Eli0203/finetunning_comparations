"""NIE-based budget allocation strategy."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NIEBudgetAllocationStrategy:
    """Allocate sample budget from NIE magnitudes with temperature softmax."""

    def __init__(
        self,
        temp_init: float = 2.0,
        temp_final: float = 0.5,
        apply_interval: int = 10,
    ) -> None:
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.apply_interval = apply_interval
        self.nie_scores: Dict[str, float] = {}
        self._warmup_state = {
            "completed": False,
            "signal": True,
            "loss": True,
            "variance": True,
            "resource": True,
        }

    def compute_nie(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        causal_paths: List[str],
    ) -> Dict[str, float]:
        """Estimate NIE per path via lightweight activation perturbation proxy."""
        if not causal_paths:
            self.nie_scores = {}
            return {}

        try:
            batch = next(iter(data_loader))
        except StopIteration:
            self.nie_scores = {p: 0.0 for p in causal_paths}
            return self.nie_scores

        if isinstance(batch, dict):
            x = batch.get("input_ids") or next(iter(batch.values()))
            if isinstance(x, torch.Tensor) and x.dtype not in (torch.float16, torch.float32, torch.float64):
                x = x.float()
        else:
            x = batch[0]

        if not isinstance(x, torch.Tensor):
            self.nie_scores = {p: 0.0 for p in causal_paths}
            return self.nie_scores

        x = x.float()
        with torch.no_grad():
            base_out = model(x)
            if isinstance(base_out, (tuple, list)):
                base_tensor = base_out[0]
            else:
                base_tensor = base_out

        scores: Dict[str, float] = {}
        for name, module in model.named_modules():
            if name not in causal_paths:
                continue
            if not hasattr(module, "weight") or module.weight is None:
                scores[name] = 0.0
                continue

            with torch.no_grad():
                original = module.weight.data.clone()
                module.weight.data.zero_()
                cf_out = model(x)
                module.weight.data.copy_(original)

                cf_tensor = cf_out[0] if isinstance(cf_out, (tuple, list)) else cf_out
                delta = (base_tensor - cf_tensor).abs().mean().item()
                scores[name] = float(delta)

        for path in causal_paths:
            scores.setdefault(path, 0.0)

        self.nie_scores = scores
        variance = float(torch.tensor(list(scores.values()), dtype=torch.float32).var(unbiased=False).item()) if scores else 0.0
        self._warmup_state["variance"] = variance > 1e-6 or all(v == 0.0 for v in scores.values())
        self._warmup_state["completed"] = all(
            bool(self._warmup_state[k]) for k in ("signal", "loss", "variance", "resource")
        )
        return scores

    def _softmax(self, values: List[float], tau: float) -> List[float]:
        if not values:
            return []
        scaled = [abs(v) / tau for v in values]
        max_v = max(scaled)
        exps = [math.exp(v - max_v) for v in scaled]
        total = sum(exps)
        return [v / total for v in exps]

    def allocate(self, causal_paths: List[str], total_budget: int) -> Dict[str, int]:
        """Allocate budget using NIE softmax, with zero-score uniform fallback."""
        if total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        if not causal_paths:
            return {}

        if not self._warmup_state.get("completed", False):
            return {p: 0 for p in causal_paths}

        scores = [self.nie_scores.get(path, 0.0) for path in causal_paths]
        if all(v == 0.0 for v in scores):
            base = total_budget // len(causal_paths)
            rem = total_budget % len(causal_paths)
            return {
                path: base + (1 if idx < rem else 0)
                for idx, path in enumerate(causal_paths)
            }

        weights = self._softmax(scores, tau=self.temp_final)
        raw = [w * total_budget for w in weights]
        alloc = [int(v) for v in raw]
        rem = total_budget - sum(alloc)
        frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - alloc[i], reverse=True)
        for i in range(rem):
            alloc[frac_idx[i]] += 1

        return {path: alloc[idx] for idx, path in enumerate(causal_paths)}
