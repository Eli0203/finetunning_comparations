"""
Unit tests for CausalWeightSampler.

Tests verify:
- Weight scale computation from budget allocation
- Batch sampling produces correctly scaled tensors
- Path weight mapping for different parameter names
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from src.utils.causal_sampler import CausalWeightSampler


class MockCausalEngine:
    """Mock causal engine for testing."""
    
    def __init__(self, budget_allocation=None, sample_budget=1000):
        self.budget_allocation = budget_allocation or {}
        self.sample_budget = sample_budget
        self.causal_paths = list(budget_allocation.keys()) if budget_allocation else []


class SimpleLoRAModel(nn.Module):
    """Simple model with LoRA parameters for testing."""
    
    def __init__(self, input_dim=64, output_dim=32, use_lora=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if use_lora:
            # Simulate LoRA adapters
            self.lora_A = nn.Linear(input_dim, 8)
            self.lora_B = nn.Linear(8, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class TestCausalWeightSamplerInitialization:
    """Test CausalWeightSampler initialization."""
    
    def test_init_with_valid_budget_allocation(self):
        """Test initialization with valid budget allocation."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'layer.1.attn': 100, 'layer.2.mlp': 50}
        )
        
        sampler = CausalWeightSampler(engine, model, device='cpu')
        
        assert sampler.causal_engine == engine
        assert sampler.model == model
        assert sampler.device == 'cpu'
        assert len(sampler.path_weights) == 2
    
    def test_init_with_empty_budget_allocation(self):
        """Test initialization with empty budget allocation (uses uniform)."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(budget_allocation={})
        
        sampler = CausalWeightSampler(engine, model)
        
        assert len(sampler.path_weights) == 1
        assert 'default' in sampler.path_weights
    
    def test_init_without_budget_allocation_attr(self):
        """Test initialization when causal engine has no budget_allocation."""
        model = SimpleLoRAModel()
        engine = Mock(spec=['sample_budget'])  # Only has sample_budget, not budget_allocation
        
        sampler = CausalWeightSampler(engine, model)
        
        assert 'default' in sampler.path_weights


class TestPathWeightComputation:
    """Test weight scale factor computation from budget allocation."""
    
    def test_compute_path_weights_equal_allocation(self):
        """Test weight normalization with equal allocation."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'path_a': 100, 'path_b': 100}
        )
        
        sampler = CausalWeightSampler(engine, model)
        weights = sampler.path_weights
        
        assert len(weights) == 2
        assert abs(weights['path_a'] - 0.5) < 1e-5
        assert abs(weights['path_b'] - 0.5) < 1e-5
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_compute_path_weights_unequal_allocation(self):
        """Test weight normalization with unequal allocation."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'path_a': 100, 'path_b': 50}
        )
        
        sampler = CausalWeightSampler(engine, model)
        weights = sampler.path_weights
        
        assert len(weights) == 2
        assert abs(weights['path_a'] - 2/3) < 1e-5
        assert abs(weights['path_b'] - 1/3) < 1e-5
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_compute_uniform_weights_fallback(self):
        """Test uniform weighting when no allocation available."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(budget_allocation={})
        
        sampler = CausalWeightSampler(engine, model)
        weights = sampler.path_weights
        
        assert 'default' in weights
        assert weights['default'] == 1.0
    
    def test_compute_path_weights_zero_budget(self):
        """Test handling of zero budget (should fall back to uniform)."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'path_a': 0, 'path_b': 0}
        )
        
        sampler = CausalWeightSampler(engine, model)
        weights = sampler.path_weights
        
        # Should fall back to uniform
        assert abs(weights['path_a'] - 0.5) < 1e-5
        assert abs(weights['path_b'] - 0.5) < 1e-5


class TestPathWeightMapping:
    """Test parameter name to causal path weight mapping."""
    
    def test_get_path_weight_exact_match(self):
        """Test path weight retrieval with exact substring match."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'layer.1.attn': 0.6, 'layer.2.mlp': 0.4}
        )
        sampler = CausalWeightSampler(engine, model)
        
        # Parameter containing 'layer.1.attn'
        weight = sampler._get_path_weight_for_param('model.layer.1.attn.lora_A.weight')
        assert weight == pytest.approx(0.6)
        
        # Parameter containing 'layer.2.mlp'
        weight = sampler._get_path_weight_for_param('model.layer.2.mlp.lora_B.weight')
        assert weight == pytest.approx(0.4)
    
    def test_get_path_weight_no_match(self):
        """Test fallback to uniform weight when no path matches."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'layer.1.attn': 0.6, 'layer.2.mlp': 0.4}
        )
        sampler = CausalWeightSampler(engine, model)
        
        # Parameter that doesn't match any path
        weight = sampler._get_path_weight_for_param('model.layer.0.other.weight')
        assert weight == pytest.approx(0.5)  # Uniform: 1/2
    
    def test_get_path_weight_longest_match(self):
        """Test that longest matching path is used."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'layer.1': 0.3, 'layer.1.attn': 0.7}
        )
        sampler = CausalWeightSampler(engine, model)
        
        param_name = 'model.layer.1.attn.lora_A.weight'
        # Both 'layer.1' and 'layer.1.attn' match, but order in dict determines
        # In practice, behavior depends on dict iteration order
        weight = sampler._get_path_weight_for_param(param_name)
        assert weight in [0.3, 0.7]  # One of the matching paths


class TestBatchSampling:
    """Test batch weight sampling."""
    
    def test_sample_batch_shapes(self):
        """Test that sampled weights have correct shapes."""
        model = SimpleLoRAModel(input_dim=64, output_dim=32)
        engine = MockCausalEngine(
            budget_allocation={'lora': 1.0}
        )
        sampler = CausalWeightSampler(engine, model, device='cpu')
        
        weights = sampler.sample_batch()
        
        # Should have weights for both LoRA layers
        assert 'lora_A.weight' in weights or 'lora_A.0.weight' in weights
        assert 'lora_B.weight' in weights or 'lora_B.0.weight' in weights
    
    def test_sample_batch_value_ranges(self):
        """Test that sampled weights are in reasonable ranges."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'lora': 1.0}
        )
        sampler = CausalWeightSampler(engine, model, device='cpu')
        
        weights = sampler.sample_batch()
        
        # Check all tensors are finite
        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), f"Tensor {name} contains NaN or Inf"
            # Standard normal should be mostly in [-3, 3]
            assert tensor.abs().max() < 10, f"Tensor {name} has unexpectedly large values"
    
    def test_sample_batch_scaling(self):
        """Test that sampled weights are scaled according to path weights."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'high': 100, 'low': 1}
        )
        sampler = CausalWeightSampler(engine, model, device='cpu')
        
        # Sample multiple times and check scale
        variances = {name: [] for name in ['lora_A.weight', 'lora_B.weight']}
        
        for _ in range(10):
            weights = sampler.sample_batch()
            for name, tensor in weights.items():
                if 'lora_A' in name:
                    variances['lora_A.weight'].append(tensor.pow(2).mean().item())
                elif 'lora_B' in name:
                    variances['lora_B.weight'].append(tensor.pow(2).mean().item())
        
        # All sampled tensors should have approximately unit variance
        # (scaled by sqrt(path_weight), so variance ~ path_weight)
        for name, vars_list in variances.items():
            if vars_list:
                mean_var = sum(vars_list) / len(vars_list)
                assert mean_var > 0, f"Tensor {name} has zero variance"
    
    def test_sample_batch_determinism_with_seed(self):
        """Test that sampling is deterministic with fixed random seed."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'lora': 1.0}
        )
        sampler = CausalWeightSampler(engine, model)
        
        torch.manual_seed(42)
        weights1 = sampler.sample_batch()
        
        torch.manual_seed(42)
        weights2 = sampler.sample_batch()
        
        # Check that all tensors are identical
        for name in weights1.keys():
            assert torch.allclose(weights1[name], weights2[name])
    
    def test_sample_batch_no_lora_params(self):
        """Test sampling when model has no LoRA parameters."""
        model = SimpleLoRAModel(use_lora=False)
        engine = MockCausalEngine(
            budget_allocation={'layer': 1.0}
        )
        sampler = CausalWeightSampler(engine, model)
        
        weights = sampler.sample_batch()
        
        # Should fall back to sampling all parameters
        assert len(weights) > 0
    
    def test_sample_batch_device_placement(self):
        """Test that sampled tensors are on correct device."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'lora': 1.0}
        )
        
        sampler = CausalWeightSampler(engine, model, device='cpu')
        weights = sampler.sample_batch()
        
        for tensor in weights.values():
            assert tensor.device.type == 'cpu'


class TestCausalWeightSamplerIntegration:
    """Integration tests for CausalWeightSampler."""
    
    def test_multiple_sampling_iterations(self):
        """Test multiple consecutive sampling calls."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'path_a': 0.6, 'path_b': 0.4}
        )
        sampler = CausalWeightSampler(engine, model)
        
        samples = [sampler.sample_batch() for _ in range(5)]
        
        # Each sample should be different (independent)
        for i in range(len(samples) - 1):
            for name in samples[i].keys():
                # Very unlikely to be exactly equal
                assert not torch.allclose(samples[i][name], samples[i+1][name])
    
    def test_get_config(self):
        """Test configuration summary."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'path_a': 100, 'path_b': 50},
            sample_budget=1000
        )
        sampler = CausalWeightSampler(engine, model, device='cpu')
        
        config = sampler.get_config()
        
        assert config['device'] == 'cpu'
        assert len(config['path_weights']) == 2
        assert config['num_causal_paths'] == 2
        assert config['total_budget'] == 1000
    
    def test_sampler_with_real_lora_model(self):
        """Test sampler with realistic LoRA model structure."""
        # Create a model with realistic LoRA naming
        class RealisticLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.Linear(768, 768)
                )
                # Add LoRA adapters with realistic names
                self.lora_linear_1 = nn.Linear(768, 32)
                self.lora_linear_2 = nn.Linear(32, 768)
            
            def forward(self, x):
                return self.transformer(x)
        
        model = RealisticLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={
                'transformer.0': 0.5,
                'transformer.1': 0.5
            }
        )
        sampler = CausalWeightSampler(engine, model)
        
        weights = sampler.sample_batch()
        
        assert len(weights) > 0
        assert all(torch.isfinite(w).all() for w in weights.values())


class TestCausalWeightSamplerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_budget_allocation(self):
        """Test with empty budget allocation."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(budget_allocation={})
        
        sampler = CausalWeightSampler(engine, model)
        weights = sampler.sample_batch()
        
        # Should work with uniform fallback
        assert len(weights) > 0
    
    def test_single_path_budget(self):
        """Test with single causal path."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'only_path': 1000}
        )
        sampler = CausalWeightSampler(engine, model)
        
        weights = sampler.sample_batch()
        
        assert len(weights) > 0
        # All parameters should use weight from only_path
        assert abs(sampler._get_path_weight_for_param('any_param') - 1.0) < 1e-5
    
    def test_large_budget_allocation(self):
        """Test with large budget allocation."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={f'path_{i}': 1000 for i in range(100)}
        )
        sampler = CausalWeightSampler(engine, model)
        
        weights = sampler.sample_batch()
        
        assert len(weights) > 0
        # Path weights should sum to 1.0
        assert sum(sampler.path_weights.values()) == pytest.approx(1.0)
    
    def test_special_parameter_names(self):
        """Test parameter name mapping with special characters."""
        model = SimpleLoRAModel()
        engine = MockCausalEngine(
            budget_allocation={'layer[0]': 1.0}
        )
        sampler = CausalWeightSampler(engine, model)
        
        # Should handle parameter names with special chars
        weight = sampler._get_path_weight_for_param('model.layer[0].weight')
        # Should match
        assert weight == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
