"""
Training integration with continuous weight application and budget monitoring.

This module provides classes for applying causal-aware weights during training
and monitoring budget consumption per causal path.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from src.utils.logger import logger


class ContinuousWeightApplier:
    """
    Applies sampled weights to model parameters during training at regular intervals.
    
    This class implements Single Responsibility Principle:
    - Applies weights to model (doesn't generate or store them)
    - Implements rate-limiting to reduce overhead
    - Handles empty buffer gracefully
    
    Design Principles:
    - Non-blocking buffer reads (never waits for weights)
    - Interval-based application (every N steps, not every step)
    - Graceful degradation (doesn't crash if buffer empty)
    - Stateless weight application (each call independent)
    """
    
    def __init__(
        self,
        buffer: Any,  # Type: DoubleBuffer
        model: nn.Module,
        device: str = 'cpu',
        apply_interval: int = 10
    ):
        """
        Initialize the weight applier.
        
        Args:
            buffer: DoubleBuffer instance from MemoryOptimizer
            model: PyTorch model containing parameters to update
            device: Device for tensor operations ('cpu' or 'cuda')
            apply_interval: Apply weights every N steps (default: 10)
            
        Raises:
            ValueError: If apply_interval <= 0
        """
        if apply_interval <= 0:
            raise ValueError(f"apply_interval must be positive, got {apply_interval}")
        
        self.buffer = buffer
        self.model = model
        self.device = device
        self.apply_interval = apply_interval
        
        # Metrics tracking
        self._times_applied = 0
        self._weight_deltas: list[float] = []
        self._last_applied_step: Optional[int] = None
        
        logger.info(
            f"ContinuousWeightApplier initialized with interval={apply_interval}, "
            f"device={device}"
        )
    
    def apply_weights(self, global_step: int) -> bool:
        """
        Apply weights from buffer to model if interval is reached.
        
        Args:
            global_step: Current training step (0-indexed)
            
        Returns:
            bool: True if weights were applied, False otherwise
        """
        # Check if we should apply weights at this step
        if not self.should_apply(global_step):
            return False
        
        try:
            # Non-blocking read from buffer
            weights_dict = self.buffer.get_latest()
            
            # Handle empty buffer
            if weights_dict is None:
                logger.debug(f"Step {global_step}: Buffer empty, skipping weight application")
                return False
            
            # Apply weights to model
            self._apply_weights_to_model(weights_dict)
            
            # Track metrics
            self._times_applied += 1
            self._last_applied_step = global_step
            
            # Log weight application with delta statistics
            if self._weight_deltas:
                avg_delta = sum(self._weight_deltas[-len(weights_dict):]) / len(weights_dict) if weights_dict else 0.0
                logger.info(f"Step {global_step}: Applied weights ({len(weights_dict)} params) | avg delta: {avg_delta:.6f}")
            else:
                logger.debug(f"Step {global_step}: Applied weights to model")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying weights at step {global_step}: {e}")
            return False
    
    def should_apply(self, step: int) -> bool:
        """
        Determine if weights should be applied at this step.
        
        Implements rate-limiting: apply at step 0, then every apply_interval steps.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            bool: True if (step % apply_interval == 0)
        """
        return (step % self.apply_interval) == 0
    
    def _apply_weights_to_model(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """
        Apply weights from dictionary to model parameters.
        
        Args:
            weights_dict: Dict mapping parameter names to tensors
        """
        for name, weight_tensor in weights_dict.items():
            try:
                # Move tensor to correct device
                weight_tensor = weight_tensor.to(self.device)
                
                # Get the parameter from model
                param_dict = dict(self.model.named_parameters())
                
                if name in param_dict:
                    param = param_dict[name]
                    
                    # Compute delta for metrics
                    delta = weight_tensor.abs().mean().item()
                    self._weight_deltas.append(delta)
                    
                    # Apply weight (add to current parameter)
                    with torch.no_grad():
                        param.add_(weight_tensor, alpha=1.0)
                else:
                    logger.debug(f"Parameter {name} not found in model, skipping")
                    
            except Exception as e:
                logger.error(f"Error applying weight {name}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cumulative metrics of weight applications.
        
        Returns:
            Dict with:
            - times_applied: Number of times weights were applied
            - last_applied_step: Step at which weights were last applied
            - mean_weight_delta: Average magnitude of weight changes
            - total_weight_delta: Sum of all weight change magnitudes
        """
        if not self._weight_deltas:
            return {
                'times_applied': self._times_applied,
                'last_applied_step': self._last_applied_step,
                'mean_weight_delta': 0.0,
                'total_weight_delta': 0.0,
            }
        
        return {
            'times_applied': self._times_applied,
            'last_applied_step': self._last_applied_step,
            'mean_weight_delta': sum(self._weight_deltas) / len(self._weight_deltas),
            'total_weight_delta': sum(self._weight_deltas),
        }
    
    def reset_metrics(self) -> None:
        """Reset accumulated metrics (e.g., at epoch boundary)."""
        self._times_applied = 0
        self._weight_deltas = []
        self._last_applied_step = None
        logger.debug("ContinuousWeightApplier metrics reset")


class TrainingBudgetMonitor:
    """
    Monitors causal budget consumption during training.
    
    Tracks how much budget is allocated vs. used per causal path,
    enabling feedback on the effectiveness of causal-aware sampling.
    
    Design Principles:
    - Per-path tracking (each path tracked independently)
    - Safe division (handles zero allocation)
    - Stateful accumulation (maintains cumulative counts)
    - Non-invasive monitoring (doesn't affect training)
    """
    
    def __init__(self, causal_engine: Any):
        """
        Initialize the budget monitor.
        
        Args:
            causal_engine: CausalMonteCLoRAEngine with budget_allocation
        """
        self.causal_engine = causal_engine
        self._consumption_tracker: Dict[str, int] = {}
        self._step_count = 0
        
        # Initialize tracker with causal paths
        if hasattr(causal_engine, 'budget_allocation') and causal_engine.budget_allocation:
            for path in causal_engine.budget_allocation:
                self._consumption_tracker[path] = 0
        
        logger.info(
            f"TrainingBudgetMonitor initialized for {len(self._consumption_tracker)} "
            f"causal paths"
        )
    
    def log_step_budget(self) -> Dict[str, Dict[str, Any]]:
        """
        Log budget allocation and consumption for this step.
        
        Returns:
            Dict mapping path names to budget info:
            {
                'path_name': {
                    'allocated': int (samples allocated),
                    'consumed': int (samples consumed so far),
                    'remaining': int (samples not yet consumed)
                },
                ...
            }
        """
        self._step_count += 1
        result: Dict[str, Dict[str, int]] = {}
        
        if not hasattr(self.causal_engine, 'budget_allocation'):
            logger.warning("Causal engine has no budget_allocation")
            return result
        
        allocation = self.causal_engine.budget_allocation
        
        for path, allocated in allocation.items():
            consumed = self._consumption_tracker.get(path, 0)
            remaining = max(0, allocated - consumed)
            
            result[path] = {
                'allocated': allocated,
                'consumed': consumed,
                'remaining': remaining,
            }
        
        logger.debug(f"Step {self._step_count}: Budget utilization: {result}")
        return result
    
    def get_budget_utilization(self) -> Dict[str, float]:
        """
        Get utilization ratio per causal path.
        
        Returns:
            Dict mapping path names to utilization ratios [0.0, 1.0]:
            {
                'path_name': ratio (consumed / allocated),
                ...
            }
            
        Note:
            Returns 0.0 for paths with zero allocation (safe division).
        """
        if not hasattr(self.causal_engine, 'budget_allocation'):
            return {}
        
        allocation = self.causal_engine.budget_allocation
        utilization: Dict[str, float] = {}
        
        for path, allocated in allocation.items():
            consumed = self._consumption_tracker.get(path, 0)
            
            # Safe division: avoid divide-by-zero
            if allocated > 0:
                ratio = min(1.0, consumed / allocated)
            else:
                ratio = 0.0
            
            utilization[path] = ratio
        
        return utilization
    
    def log_weight_application(self, path: str, count: int = 1) -> None:
        """
        Log weight application for a causal path.
        
        Args:
            path: Causal path name
            count: Number of samples to increment (default: 1)
        """
        if path not in self._consumption_tracker:
            self._consumption_tracker[path] = 0
        
        self._consumption_tracker[path] += count
    
    def reset(self) -> None:
        """Reset consumption tracker (e.g., at epoch boundary)."""
        for path in self._consumption_tracker:
            self._consumption_tracker[path] = 0
        self._step_count = 0
        logger.debug("TrainingBudgetMonitor reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of budget monitoring.
        
        Returns:
            Dict with:
            - total_steps: Number of steps monitored
            - total_budget: Total budget across all paths
            - total_consumed: Total consumption across all paths
            - utilization: Per-path utilization ratios
        """
        allocation = self.causal_engine.budget_allocation if hasattr(self.causal_engine, 'budget_allocation') else {}
        total_budget = sum(allocation.values()) if allocation else 0
        total_consumed = sum(self._consumption_tracker.values())
        utilization = self.get_budget_utilization()
        
        return {
            'total_steps': self._step_count,
            'total_budget': total_budget,
            'total_consumed': total_consumed,
            'utilization': utilization,
        }
