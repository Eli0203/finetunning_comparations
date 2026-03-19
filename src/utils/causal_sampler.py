"""
Causal-aware weight sampling for LoRA fine-tuning.

This module provides intelligent weight generation informed by causal budget allocation,
replacing random weight generation in the async sampling pipeline.
"""

import torch
import torch.nn as nn
from typing import Any, Dict
from src.utils.logger import logger


class CausalWeightSampler:
    """
    Generates weights for LoRA adapters informed by causal budget allocation.
    
    This class implements Single Responsibility Principle:
    - Takes causal decisions (paths, budgets) from causal engine
    - Produces weight tensors scaled by causal importance
    - Does NOT implement: training logic, data loading, multiprocessing
    
    Design Principles:
    - Dependency injection of causal_engine (enables testing with mocks)
    - Immutable initialization (config set once at construction)
    - Stateless sampling (each call is independent)
    - OOP design (class-based, composable)
    """

    def __init__(
        self,
        causal_engine: Any,  # Type: CausalMonteCLoRAEngine
        model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize the causal weight sampler.
        
        Args:
            causal_engine: CausalMonteCLoRAEngine instance with budget allocation
            model: PyTorch model containing LoRA adapters
            device: Device for tensor operations ('cpu' or 'cuda')
            
        Raises:
            ValueError: If causal_engine has no budget allocation
        """
        self.causal_engine = causal_engine
        self.model = model
        self.device = device
        
        # Validate that causal engine has computed allocation
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
    
    def sample_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Sample random weights scaled by causal importance.
        
        For each LoRA adapter in the model:
        1. Identify which causal path it belongs to
        2. Get the scale factor for that path
        3. Generate random tensor: N(0, scale^2)
        4. Return as state_dict
        
        Args:
            batch_size: Number of independent samples (currently unused, generates 1)
            
        Returns:
            Dict mapping parameter names to sampled weight tensors.
            Format compatible with model.load_state_dict()
        """
        state_dict = {}
        magnitude_stats: Dict[str, float] = {}
        
        for name, param in self.model.named_parameters():
            # Only sample LoRA parameters (skip non-LoRA params)
            if 'lora' not in name.lower():
                continue
            
            # Get the causal path weight for this parameter
            path_weight = self._get_path_weight_for_param(name)
            
            # Sample: variance = path_weight, so std = sqrt(path_weight)
            std = torch.sqrt(torch.tensor(path_weight, dtype=torch.float32))
            
            # Generate random tensor on the correct device
            sampled = std * torch.randn_like(param, device=self.device)
            
            # Ensure tensor is on correct device
            state_dict[name] = sampled.to(self.device)
            
            # Track magnitude for observability
            magnitude = sampled.abs().mean().item()
            magnitude_stats[name] = magnitude
        
        if not state_dict:
            logger.warning("No LoRA parameters found in model. Sampling all parameters.")
            for name, param in self.model.named_parameters():
                state_dict[name] = torch.randn_like(param, device=self.device)
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
            'total_budget': self.causal_engine.sample_budget if hasattr(self.causal_engine, 'sample_budget') else None
        }
