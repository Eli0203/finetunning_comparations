"""
Causal-aware weight sampling for LoRA fine-tuning.

This module provides intelligent weight generation informed by causal budget allocation,
replacing random weight generation in the async sampling pipeline.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from src.utils.logger import logger


class CausalWeightSampler:
    """
    Generates weights for LoRA adapters informed by causal budget allocation.

    Single Responsibility:
    - Translate causal decisions (paths, budgets) into weight tensors
    - Does NOT implement: training logic, data loading, multiprocessing

    Design Principles:
    - Dependency injection of causal_engine (enables testing with mocks)
    - Spawn-safe: stores only CPU-compatible metadata (shapes/dtypes) instead
      of the live model.  CUDA tensors are never held after __init__.
    - Always produces CPU tensors; consumers must move them to the target device.
    - Pickle protocol (__getstate__/__setstate__) excludes the causal_engine
      (which may hold CUDA model refs) so the sampler can be safely transmitted
      to a spawned worker process.
    """

    def __init__(
        self,
        causal_engine: Any,  # Type: CausalMonteCLoRAEngine
        model: nn.Module,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize the causal weight sampler.

        Args:
            causal_engine: CausalMonteCLoRAEngine instance with budget allocation.
            model: PyTorch model containing LoRA adapters.  Only parameter
                   metadata (shapes, dtypes) is retained; the live model tensor
                   data is NOT stored, keeping this object spawn-safe.
            device: Target device hint stored for external callers — workers
                    always generate CPU tensors regardless of this value.
        """
        self.causal_engine: Optional[Any] = causal_engine
        self.device = device

        # --- Spawn-safe parameter metadata (no CUDA tensor references) ---
        # Storing only shape/dtype lets the worker generate correctly-shaped
        # CPU tensors without holding a reference to the live CUDA model.
        self._param_specs: Dict[str, Tuple[torch.Size, torch.dtype]] = {
            name: (param.shape, param.dtype)
            for name, param in model.named_parameters()
        }
        self._lora_param_specs: Dict[str, Tuple[torch.Size, torch.dtype]] = {
            name: spec
            for name, spec in self._param_specs.items()
            if 'lora' in name.lower()
        }

        # Initialise path weights from current allocation (may be empty here
        # if allocate_budget() has not been called yet; call refresh_path_weights
        # after allocation to correct the weighting).
        if not hasattr(causal_engine, 'budget_allocation') or not causal_engine.budget_allocation:
            logger.warning("Causal engine has no budget allocation. Using uniform weighting.")
            self.path_weights = self._compute_uniform_weights()
        else:
            self.path_weights = self._compute_path_weights()

        logger.info(f"CausalWeightSampler initialized with paths: {list(self.path_weights.keys())}")
    
    def _compute_path_weights(self) -> Dict[str, float]:
        """
        Derive weight scale factors from causal budget allocation.
        
        Maps the budget allocation dict (path_name → sample_count) to normalized weights
        (path_name → scale_factor in [0, 1]).
        
        Example:
            Input: {'layer.1.attn': 100, 'layer.2.mlp': 50}
            Total: 150
            Output: {'layer.1.attn': 0.667, 'layer.2.mlp': 0.333}
        
        Returns:
            Dict mapping causal path names to weight scale factors
        """
        allocation = self.causal_engine.budget_allocation
        
        if not allocation:
            return {}
        
        total = sum(allocation.values())
        if total == 0:
            # All paths have zero budget - use uniform as fallback
            return {path: 1.0 / len(allocation) for path in allocation}
        
        weights = {path: count / total for path, count in allocation.items()}
        logger.debug(f"Computed path weights: {weights}")
        
        return weights
    
    def _compute_uniform_weights(self) -> Dict[str, float]:
        """
        Compute uniform weights when no causal allocation available.
        
        Used as fallback when causal engine hasn't run identify_causal_paths/allocate_budget.
        
        Returns:
            Dict with single entry for uniform weighting
        """
        logger.debug("Using uniform weighting (no causal allocation available)")
        return {"default": 1.0}
    
    # ------------------------------------------------------------------
    # Pickle protocol — exclude causal_engine (may hold CUDA model refs)
    # so the sampler can be safely transmitted to a spawned worker process.
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state (causal_engine excluded; not needed in worker)."""
        state = self.__dict__.copy()
        state.pop('causal_engine', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore from pickled state; causal_engine unavailable in worker process."""
        self.__dict__.update(state)
        self.causal_engine = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_path_weights(self) -> None:
        """Re-read budget allocation from causal_engine and update path weights.

        Call this after ``causal_engine.allocate_budget()`` to ensure the sampler
        uses the correct causal-informed weights.  This is required when the
        sampler is constructed before ``prepare()`` is called on the orchestrator.
        """
        if self.causal_engine is None:
            logger.warning("refresh_path_weights: causal_engine not available; skipping.")
            return
        if hasattr(self.causal_engine, 'budget_allocation') and self.causal_engine.budget_allocation:
            self.path_weights = self._compute_path_weights()
            logger.info(f"CausalWeightSampler path weights refreshed: {self.path_weights}")
        else:
            logger.warning("refresh_path_weights: causal_engine still has no budget allocation.")

    def sample_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Sample random weights scaled by causal importance.

        For each LoRA adapter:
        1. Identify which causal path it belongs to.
        2. Get the scale factor for that path.
        3. Generate a random CPU tensor: N(0, scale²).
        4. Return as a state-dict-compatible mapping.

        Tensors are **always on CPU** so they can be safely transmitted through
        multiprocessing Manager proxies.  ``ContinuousWeightApplier`` is
        responsible for moving them to the target device on application.

        Args:
            batch_size: Number of independent samples (currently unused; generates 1).

        Returns:
            Dict mapping parameter names to sampled weight tensors (CPU).
        """
        state_dict: Dict[str, torch.Tensor] = {}
        magnitude_stats: Dict[str, float] = {}

        specs = self._lora_param_specs if self._lora_param_specs else self._param_specs
        for name, (shape, dtype) in specs.items():
            path_weight = self._get_path_weight_for_param(name)
            std = torch.sqrt(torch.tensor(path_weight, dtype=torch.float32))
            # Always CPU — no CUDA tensor crosses the process boundary
            sampled = std * torch.randn(shape, dtype=dtype)
            if sampled.device.type != 'cpu':
                logger.warning(f"Tensor {name} sampled on {sampled.device}; moving to CPU")
                sampled = sampled.cpu()
            state_dict[name] = sampled
            magnitude_stats[name] = sampled.abs().mean().item()

        if not state_dict:
            logger.warning("No LoRA parameters found. Sampling all parameters.")
            for name, (shape, dtype) in self._param_specs.items():
                sampled = torch.randn(shape, dtype=dtype)
                if sampled.device.type != 'cpu':
                    sampled = sampled.cpu()
                state_dict[name] = sampled
                magnitude_stats[name] = state_dict[name].abs().mean().item()
        
        # Log per-parameter weight magnitudes for observability
        avg_magnitude = sum(magnitude_stats.values()) / len(magnitude_stats) if magnitude_stats else 0.0
        logger.debug(f"Sampled weights for {len(state_dict)} parameters | avg magnitude: {avg_magnitude:.6f}")
        
        # Log detailed magnitude breakdown for high-variance parameters
        for name, mag in sorted(magnitude_stats.items(), key=lambda x: x[1], reverse=True)[:3]:
            logger.debug(f"  - {name}: magnitude={mag:.6f}")
        
        return state_dict
    
    def _get_path_weight_for_param(self, param_name: str) -> float:
        """
        Map parameter name to its causal path weight.
        
        Implements a simple string-matching heuristic: if the causal path substring
        appears in the parameter name, use that path's weight. Falls back to uniform
        weighting if no match found.
        
        Args:
            param_name: Full parameter name (e.g., 'model.layers.0.self_attn.lora_A')
            
        Returns:
            Weight scale factor in (0, 1]
            
        Example:
            path_weights = {'layer.1.attn': 0.6, 'layer.2.mlp': 0.4}
            param_name = 'model.layer.1.self_attn.lora_A'
            → looks for 'layer.1.attn' in param_name, finds 'layer.1'
            → returns 0.6
        """
        # Exact match first (preferred)
        for path, weight in self.path_weights.items():
            if path in param_name:
                return weight
        
        # Fallback: uniform weighting
        if self.path_weights:
            uniform_weight = 1.0 / len(self.path_weights)
        else:
            uniform_weight = 1.0
        
        logger.debug(f"No path match for {param_name}, using uniform weight {uniform_weight}")
        return uniform_weight
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration summary of sampler.
        
        Returns:
            Dict with sampler configuration and current state
        """
        return {
            'device': self.device,
            'path_weights': self.path_weights,
            'num_causal_paths': len(self.path_weights),
            'num_param_specs': len(self._param_specs),
            'total_budget': (
                self.causal_engine.sample_budget
                if self.causal_engine is not None and hasattr(self.causal_engine, 'sample_budget')
                else None
            ),
        }
