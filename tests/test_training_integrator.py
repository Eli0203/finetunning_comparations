"""
Unit tests for training integration modules.

Tests verify:
- Weight application at correct intervals
- Rate-limiting logic
- Budget monitoring and utilization tracking
- Safe failure when buffer is empty
- Metrics accumulation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from src.utils.training_integrator import ContinuousWeightApplier, TrainingBudgetMonitor


class SimpleDummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=64, output_dim=32):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class MockDoubleBuffer:
    """Mock DoubleBuffer for testing."""
    
    def __init__(self, return_weights=True):
        self.return_weights = return_weights
        self.put_count = 0
    
    def put(self, item):
        self.put_count += 1
    
    def get_latest(self):
        if not self.return_weights:
            return None
        
        # Return a dict of random weights
        return {
            'linear1.weight': torch.randn(32, 64),
            'linear1.bias': torch.randn(32),
            'linear2.weight': torch.randn(10, 32),
            'linear2.bias': torch.randn(10),
        }


class MockCausalEngine:
    """Mock causal engine for testing."""
    
    def __init__(self, budget_allocation=None):
        self.budget_allocation = budget_allocation or {'path_a': 100, 'path_b': 50}
        self.sample_budget = 1000


# ============================================================================
# Tests for ContinuousWeightApplier
# ============================================================================

class TestContinuousWeightApplierInitialization:
    """Test ContinuousWeightApplier initialization."""
    
    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        
        applier = ContinuousWeightApplier(buffer, model, device='cpu', apply_interval=10)
        
        assert applier.buffer == buffer
        assert applier.model == model
        assert applier.device == 'cpu'
        assert applier.apply_interval == 10
    
    def test_init_with_invalid_interval(self):
        """Test initialization with invalid apply_interval."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        
        with pytest.raises(ValueError):
            ContinuousWeightApplier(buffer, model, apply_interval=0)
        
        with pytest.raises(ValueError):
            ContinuousWeightApplier(buffer, model, apply_interval=-10)
    
    def test_init_metrics_tracking(self):
        """Test that metrics are initialized."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        
        applier = ContinuousWeightApplier(buffer, model)
        
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 0
        assert metrics['last_applied_step'] is None
        assert metrics['mean_weight_delta'] == 0.0


class TestWeightApplicationLogic:
    """Test weight application core logic."""
    
    def test_apply_weights_returns_bool(self):
        """Test that apply_weights returns boolean."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        result = applier.apply_weights(0)
        
        assert isinstance(result, bool)
        assert result is True
    
    def test_apply_weights_at_interval(self):
        """Test that weights are applied at correct intervals."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=5)
        
        # Step 0: should apply
        assert applier.apply_weights(0) is True
        
        # Steps 1-4: should not apply
        for step in range(1, 5):
            assert applier.apply_weights(step) is False
        
        # Step 5: should apply
        assert applier.apply_weights(5) is True
    
    def test_apply_weights_metrics_updated(self):
        """Test that metrics are updated when weights applied."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Apply weights
        applier.apply_weights(0)
        
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 1
        assert metrics['last_applied_step'] == 0
        assert metrics['mean_weight_delta'] > 0
    
    def test_apply_weights_empty_buffer(self):
        """Test handling of empty buffer."""
        buffer = MockDoubleBuffer(return_weights=False)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Should handle gracefully
        result = applier.apply_weights(0)
        
        assert result is False
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 0


class TestRateLimitingLogic:
    """Test rate-limiting functionality."""
    
    def test_should_apply_interval_zero(self):
        """Test should_apply at step 0."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=10)
        
        assert applier.should_apply(0) is True
    
    def test_should_apply_interval_boundary(self):
        """Test should_apply at interval boundaries."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=5)
        
        assert applier.should_apply(0) is True
        assert applier.should_apply(5) is True
        assert applier.should_apply(10) is True
        assert applier.should_apply(15) is True
    
    def test_should_apply_non_boundary(self):
        """Test should_apply between boundaries."""
        buffer = MockDoubleBuffer()
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=5)
        
        assert applier.should_apply(1) is False
        assert applier.should_apply(2) is False
        assert applier.should_apply(3) is False
        assert applier.should_apply(4) is False
        assert applier.should_apply(6) is False
    
    def test_apply_interval_one(self):
        """Test with apply_interval=1 (apply every step)."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Should apply at every step
        for step in range(10):
            assert applier.should_apply(step) is True


class TestMetricsTracking:
    """Test metrics tracking functionality."""
    
    def test_metrics_format(self):
        """Test metrics return format."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        applier.apply_weights(0)
        metrics = applier.get_metrics()
        
        assert 'times_applied' in metrics
        assert 'last_applied_step' in metrics
        assert 'mean_weight_delta' in metrics
        assert 'total_weight_delta' in metrics
    
    def test_metrics_accumulation(self):
        """Test metrics accumulate across multiple applications."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Apply weights multiple times
        for step in range(5):
            applier.apply_weights(step)
        
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 5
        assert metrics['last_applied_step'] == 4
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Apply weights
        applier.apply_weights(0)
        applier.apply_weights(1)
        
        # Reset
        applier.reset_metrics()
        
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 0
        assert metrics['last_applied_step'] is None
        assert metrics['mean_weight_delta'] == 0.0


# ============================================================================
# Tests for TrainingBudgetMonitor
# ============================================================================

class TestTrainingBudgetMonitorInitialization:
    """Test TrainingBudgetMonitor initialization."""
    
    def test_init_with_causal_engine(self):
        """Test initialization with causal engine."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        
        monitor = TrainingBudgetMonitor(engine)
        
        assert monitor.causal_engine == engine
        assert len(monitor._consumption_tracker) == 2
    
    def test_init_with_empty_budget(self):
        """Test initialization with empty budget allocation."""
        # Create engine with no budget_allocation attribute
        engine = Mock(spec=['sample_budget'])
        engine.sample_budget = 1000
        
        monitor = TrainingBudgetMonitor(engine)
        
        assert len(monitor._consumption_tracker) == 0


class TestBudgetLogging:
    """Test budget logging functionality."""
    
    def test_log_step_budget_format(self):
        """Test log_step_budget return format."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        monitor = TrainingBudgetMonitor(engine)
        
        budget = monitor.log_step_budget()
        
        assert isinstance(budget, dict)
        assert 'path_a' in budget
        assert 'path_b' in budget
        assert 'allocated' in budget['path_a']
        assert 'consumed' in budget['path_a']
        assert 'remaining' in budget['path_a']
    
    def test_log_step_budget_values(self):
        """Test log_step_budget correct values."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        monitor = TrainingBudgetMonitor(engine)
        
        # Log initial budget
        budget = monitor.log_step_budget()
        assert budget['path_a']['allocated'] == 100
        assert budget['path_a']['consumed'] == 0
        assert budget['path_a']['remaining'] == 100
        
        # Log after consumption
        monitor.log_weight_application('path_a', 30)
        budget = monitor.log_step_budget()
        assert budget['path_a']['consumed'] == 30
        assert budget['path_a']['remaining'] == 70
    
    def test_log_weight_application(self):
        """Test weight application logging."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100})
        monitor = TrainingBudgetMonitor(engine)
        
        monitor.log_weight_application('path_a', 20)
        monitor.log_weight_application('path_a', 15)
        
        budget = monitor.log_step_budget()
        assert budget['path_a']['consumed'] == 35


class TestUtilizationCalculation:
    """Test budget utilization calculation."""
    
    def test_get_budget_utilization_format(self):
        """Test utilization return format."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        monitor = TrainingBudgetMonitor(engine)
        
        utilization = monitor.get_budget_utilization()
        
        assert isinstance(utilization, dict)
        assert 'path_a' in utilization
        assert 'path_b' in utilization
    
    def test_get_budget_utilization_ranges(self):
        """Test utilization values are in [0, 1]."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100})
        monitor = TrainingBudgetMonitor(engine)
        
        # Test 0% utilization
        util = monitor.get_budget_utilization()
        assert util['path_a'] == 0.0
        
        # Test 50% utilization
        monitor.log_weight_application('path_a', 50)
        util = monitor.get_budget_utilization()
        assert util['path_a'] == 0.5
        
        # Test 100% utilization
        monitor.log_weight_application('path_a', 50)
        util = monitor.get_budget_utilization()
        assert util['path_a'] == 1.0
        
        # Test >100% utilization (capped at 1.0)
        monitor.log_weight_application('path_a', 10)
        util = monitor.get_budget_utilization()
        assert util['path_a'] == 1.0
    
    def test_zero_allocation_safety(self):
        """Test safe division with zero allocation."""
        engine = MockCausalEngine(budget_allocation={'path_a': 0})
        monitor = TrainingBudgetMonitor(engine)
        
        # Should not raise division by zero
        util = monitor.get_budget_utilization()
        
        assert util['path_a'] == 0.0


class TestMonitorReset:
    """Test budget monitor reset functionality."""
    
    def test_reset_consumption(self):
        """Test reset clears consumption."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100})
        monitor = TrainingBudgetMonitor(engine)
        
        monitor.log_weight_application('path_a', 50)
        monitor.reset()
        
        budget = monitor.log_step_budget()
        assert budget['path_a']['consumed'] == 0
    
    def test_reset_step_count(self):
        """Test reset clears step count."""
        engine = MockCausalEngine()
        monitor = TrainingBudgetMonitor(engine)
        
        monitor.log_step_budget()
        monitor.log_step_budget()
        monitor.reset()
        
        assert monitor._step_count == 0


class TestMonitorSummary:
    """Test monitor summary functionality."""
    
    def test_get_summary_format(self):
        """Test summary return format."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        monitor = TrainingBudgetMonitor(engine)
        
        summary = monitor.get_summary()
        
        assert 'total_steps' in summary
        assert 'total_budget' in summary
        assert 'total_consumed' in summary
        assert 'utilization' in summary
    
    def test_get_summary_values(self):
        """Test summary contains correct values."""
        engine = MockCausalEngine(budget_allocation={'path_a': 100, 'path_b': 50})
        monitor = TrainingBudgetMonitor(engine)
        
        monitor.log_step_budget()
        monitor.log_weight_application('path_a', 30)
        monitor.log_weight_application('path_b', 20)
        
        summary = monitor.get_summary()
        
        assert summary['total_budget'] == 150
        assert summary['total_consumed'] == 50


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for applier and monitor together."""
    
    def test_applier_and_monitor_together(self):
        """Test applier and monitor work together."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        engine = MockCausalEngine(budget_allocation={'linear1': 100, 'linear2': 50})
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=5)
        monitor = TrainingBudgetMonitor(engine)
        
        # Simulate training loop
        for step in range(10):
            # Apply weights
            if applier.apply_weights(step):
                monitor.log_weight_application('linear1', 1)
            
            # Log budget
            budget = monitor.log_step_budget()
            assert budget is not None
    
    def test_simulated_training_loop(self):
        """Test simulated training loop."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        engine = MockCausalEngine(budget_allocation={'path': 100})
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=3)
        monitor = TrainingBudgetMonitor(engine)
        
        applied_count = 0
        for step in range(10):
            if applier.apply_weights(step):
                applied_count += 1
                monitor.log_weight_application('path', 1)
        
        # Should apply at steps 0, 3, 6, 9
        assert applied_count == 4
        
        # Check metrics
        metrics = applier.get_metrics()
        assert metrics['times_applied'] == 4
        
        # Check budget
        util = monitor.get_budget_utilization()
        assert util['path'] == 0.04  # 4/100


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_large_apply_interval(self):
        """Test with very large apply interval."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1000)
        
        # Should only apply at step 0
        assert applier.apply_weights(0) is True
        for step in range(1, 100):
            assert applier.apply_weights(step) is False
    
    def test_missing_causal_engine_attribute(self):
        """Test when causal engine lacks budget_allocation."""
        engine = Mock(spec=[])  # Empty spec
        monitor = TrainingBudgetMonitor(engine)
        
        # Should handle gracefully
        budget = monitor.log_step_budget()
        assert isinstance(budget, dict)
    
    def test_weights_with_different_devices(self):
        """Test applying weights with device handling."""
        buffer = MockDoubleBuffer(return_weights=True)
        model = SimpleDummyModel()
        
        # Create applier for CPU
        applier = ContinuousWeightApplier(buffer, model, device='cpu', apply_interval=1)
        
        # Should apply without error
        result = applier.apply_weights(0)
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
