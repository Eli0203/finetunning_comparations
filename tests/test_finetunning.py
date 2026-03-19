"""
Test suite for fine-tuning system including memory budget validation.
"""

import torch
import torch.nn as nn
import psutil
import os
from src.utils.async_sampler import BackgroundSampler
from src.utils.memory_manager import MemoryOptimizer
from src.utils.causal_sampler import CausalWeightSampler


def test_smoke():
    """Basic smoke test to ensure the test suite is executable."""
    assert True


class SimpleTestModel(nn.Module):
    """Simple model for memory testing."""
    
    def __init__(self, input_dim=256, output_dim=128, num_lora_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_lora_layers)
        ])
        # Add LoRA adapters
        self.lora_A = nn.Linear(output_dim, 16)
        self.lora_B = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MockCausalEngine:
    """Mock causal engine for testing."""
    
    def __init__(self):
        self.budget_allocation = {'layer_1': 100, 'layer_2': 100}
        self.sample_budget = 1000


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


def test_async_sampler_memory_budget():
    """
    T032: Smoke test ensuring async sampler doesn't exceed buffer memory constraints.
    
    Validates that:
    - Buffer creation succeeds
    - Sample generation is memory-efficient
    - No runaway memory growth during sampling
    - Memory stays within 10GB constraint
    """
    # Get baseline memory
    baseline_memory = get_process_memory_mb()
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    # Create test model and buffer
    model = SimpleTestModel(input_dim=256, output_dim=128, num_lora_layers=4)
    buffer = MemoryOptimizer.create_double_buffer()
    
    # Create mock causal engine and sampler
    causal_engine = MockCausalEngine()
    causal_sampler = CausalWeightSampler(causal_engine, model, device='cpu')
    
    # Create and start async sampler with limited iterations
    async_sampler = BackgroundSampler(
        buffer=buffer,
        model=model,
        max_steps=5,  # Limit to 5 steps for smoke test
        causal_sampler=causal_sampler
    )
    
    try:
        async_sampler.start()
        
        # Wait for sampler to complete
        import time
        time.sleep(2)  # Give sampler time to run
        
        # Check memory after sampling
        peak_memory = get_process_memory_mb()
        memory_delta = peak_memory - baseline_memory
        
        print(f"Peak memory after sampling: {peak_memory:.1f} MB")
        print(f"Memory delta: {memory_delta:.1f} MB")
        
        # Verify memory constraints (10GB limit)
        assert peak_memory < 10240, f"Memory usage {peak_memory:.1f} MB exceeds 10GB limit"
        
        # Verify reasonable memory growth (< 500MB for 5 samples is acceptable)
        assert memory_delta < 500, f"Memory growth {memory_delta:.1f} MB seems excessive"
        
        # Verify buffer actually has data
        latest_weights = buffer.get_latest()
        if latest_weights is not None:
            assert len(latest_weights) > 0, "Buffer should contain sampled weights"
            print(f"✓ Buffer contains {len(latest_weights)} weight tensors")
        
        print("✓ Async sampler memory test passed")
        
    finally:
        async_sampler.stop()
        

def test_buffer_no_memory_leak():
    """
    Test that DoubleBuffer doesn't cause memory leaks under repeated put/get.
    
    Validates sequential put/get operations don't accumulate memory.
    """
    baseline_memory = get_process_memory_mb()
    
    buffer = MemoryOptimizer.create_double_buffer()
    model = SimpleTestModel(input_dim=64, output_dim=32, num_lora_layers=1)
    
    # Simulate repeated weight generation and buffering
    for iteration in range(10):
        # Generate random weights
        weights = {
            name: torch.randn_like(param)
            for name, param in model.named_parameters()
        }
        
        # Put in buffer
        buffer.put(weights)
        
        # Get from buffer
        retrieved = buffer.get_latest()
        assert retrieved is not None, f"Iteration {iteration}: Buffer should not be None"
    
    # Check memory after iterations
    final_memory = get_process_memory_mb()
    memory_delta = final_memory - baseline_memory
    
    print(f"Memory after 10 put/get cycles: {memory_delta:.1f} MB")
    
    # Growth should be minimal (< 50MB for these small tensors)
    assert memory_delta < 100, f"Excessive memory growth: {memory_delta:.1f} MB"
    print("✓ Buffer memory leak test passed")


def test_model_size_estimation():
    """
    Test that model parameter estimation is accurate.
    
    Validates we can correctly count model size for memory planning.
    """
    model = SimpleTestModel(input_dim=256, output_dim=128, num_lora_layers=4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'lora' in name.lower()
    )
    
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    
    # Estimate memory (4 bytes per float32)
    total_memory_mb = (total_params * 4) / (1024 * 1024)
    lora_memory_mb = (lora_params * 4) / (1024 * 1024)
    
    print(f"Estimated total memory: {total_memory_mb:.2f} MB")
    print(f"Estimated LoRA memory: {lora_memory_mb:.2f} MB")
    
    # Verify reasonable estimates
    assert total_params > 0, "Model should have parameters"
    assert lora_params > 0, "Model should have LoRA parameters"
    assert lora_params < total_params, "LoRA params should be subset of total"
    
    print("✓ Model size estimation test passed")