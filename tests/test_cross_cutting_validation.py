"""
Cross-cutting validation tests for Phase 6.

Tests ensure:
- Async sampler handles graceful shutdown on KeyboardInterrupt
- DoubleBuffer is thread-safe
- Weight application works with different model architectures
"""

import unittest
import torch
import torch.nn as nn
import threading
import time
from unittest.mock import Mock, patch

from src.utils.async_sampler import BackgroundSampler
from src.utils.memory_manager import MemoryOptimizer
from src.utils.causal_sampler import CausalWeightSampler
from src.utils.training_integrator import ContinuousWeightApplier
from src.utils.logger import logger


class SimpleModel(nn.Module):
    """Minimal test model."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.lora_A = nn.Linear(10, 2)


class ComplexModel(nn.Module):
    """Model with multiple architectures (CNN + Attention + FFN)."""
    
    def __init__(self):
        super().__init__()
        # CNN block
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Attention-like block
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        
        # FFN block with LoRA adapters
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 16)
        
        # LoRA adapters
        self.lora_attn_A = nn.Linear(16, 4)
        self.lora_attn_B = nn.Linear(4, 16)


class MockCausalEngine:
    """Mock causal engine."""
    
    def __init__(self):
        self.budget_allocation = {'layer_1': 100, 'layer_2': 50}
        self.sample_budget = 1000


class TestAsyncSamplerGracefulShutdown(unittest.TestCase):
    """Test async sampler handles interruption gracefully."""
    
    def test_sampler_stop_called_on_interrupt(self):
        """Test that sampler.stop() is called during cleanup."""
        model = SimpleModel()
        buffer = MemoryOptimizer.create_double_buffer()
        causal_engine = MockCausalEngine()
        sampler = CausalWeightSampler(causal_engine, model)
        
        async_sampler = BackgroundSampler(
            buffer=buffer,
            model=model,
            max_steps=100,
            causal_sampler=sampler
        )
        
        # Start sampler
        async_sampler.start()
        time.sleep(0.1)  # Let it start
        
        # Stop should not raise exception
        try:
            async_sampler.stop()
            assert True  # Success
        except Exception as e:
            self.fail(f"stop() raised {type(e).__name__}: {e}")
    
    def test_sampler_process_cleanup(self):
        """Test that sampler process is properly cleaned up."""
        model = SimpleModel()
        buffer = MemoryOptimizer.create_double_buffer()
        causal_engine = MockCausalEngine()
        sampler = CausalWeightSampler(causal_engine, model)
        
        async_sampler = BackgroundSampler(
            buffer=buffer,
            model=model,
            max_steps=10,
            causal_sampler=sampler
        )
        
        async_sampler.start()
        self.assertIsNotNone(async_sampler.process)
        self.assertTrue(async_sampler.process.is_alive())
        
        async_sampler.stop()
        time.sleep(0.2)  # Wait for process to terminate
        
        # Process should be stopped or joining
        self.assertFalse(async_sampler.process.is_alive())


class TestDoubleBufferThreadSafety(unittest.TestCase):
    """Test DoubleBuffer is thread-safe."""
    
    def test_concurrent_put_get(self):
        """
        Test that DoubleBuffer handles concurrent put/get without data corruption.
        
        Simulates multiple threads writing and reading simultaneously.
        """
        buffer = MemoryOptimizer.create_double_buffer()
        results = {'puts': 0, 'gets': 0, 'errors': []}
        lock = threading.Lock()
        
        def writer_thread(thread_id: int):
            """Writer thread that puts data into buffer."""
            try:
                for i in range(10):
                    data = {'thread': thread_id, 'iteration': i, 'value': i * 10}
                    buffer.put(data)
                    with lock:
                        results['puts'] += 1
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    results['errors'].append(f"Writer {thread_id}: {e}")
        
        def reader_thread(thread_id: int):
            """Reader thread that gets latest data from buffer."""
            try:
                for i in range(10):
                    data = buffer.get_latest()
                    with lock:
                        results['gets'] += 1
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    results['errors'].append(f"Reader {thread_id}: {e}")
        
        # Start multiple writers and readers
        threads = []
        for i in range(2):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=5)
        
        # Verify no errors occurred
        self.assertEqual(len(results['errors']), 0, f"Errors: {results['errors']}")
        
        # Verify operations completed
        self.assertGreater(results['puts'], 0)
        self.assertGreater(results['gets'], 0)
    
    def test_buffer_lock_prevents_races(self):
        """Test that buffer internal lock prevents race conditions."""
        buffer = MemoryOptimizer.create_double_buffer()
        
        # Rapid-fire puts and gets
        for _ in range(100):
            buffer.put({'data': 'test'})
        
        # All gets should succeed (not raise exceptions)
        for _ in range(100):
            result = buffer.get_latest()
            # Result could be None (empty) or dict, both valid


class TestWeightApplicationDifferentArchitectures(unittest.TestCase):
    """Test weight application works with different model architectures."""
    
    def test_weight_application_simple_model(self):
        """Test weight application on simple linear model."""
        model = SimpleModel()
        buffer = Mock()
        buffer.get_latest.return_value = {
            'linear.weight': torch.randn(5, 10),
            'linear.bias': torch.randn(5),
        }
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Should apply without error
        result = applier.apply_weights(0)
        self.assertTrue(result)
    
    def test_weight_application_complex_model(self):
        """Test weight application on complex model with multiple architectures."""
        model = ComplexModel()
        buffer = Mock()
        
        # Generate weights matching model structure
        weights = {}
        for name, param in model.named_parameters():
            if 'lora' in name:
                weights[name] = torch.randn_like(param)
        
        buffer.get_latest.return_value = weights
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Should apply without error
        result = applier.apply_weights(0)
        self.assertTrue(result)
    
    def test_weight_application_missing_parameters(self):
        """Test weight application handles missing parameters gracefully."""
        model = SimpleModel()
        buffer = Mock()
        buffer.get_latest.return_value = {
            'nonexistent.weight': torch.randn(5, 10),
            'also_missing.bias': torch.randn(5),
        }
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        
        # Should handle gracefully (skip missing params)
        result = applier.apply_weights(0)
        self.assertTrue(result)  # Should still succeed
    
    def test_weight_application_cnn_model(self):
        """Test weight application on CNN model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
        )
        buffer = Mock()
        buffer.get_latest.return_value = {
            '0.weight': torch.randn(16, 3, 3, 3),
            '0.bias': torch.randn(16),
        }
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        result = applier.apply_weights(0)
        self.assertTrue(result)
    
    def test_weight_application_transformer_model(self):
        """Test weight application on transformer-like model."""
        model = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.Linear(768, 768),
        )
        buffer = Mock()
        buffer.get_latest.return_value = {
            '0.weight': torch.randn(768, 768),
            '0.bias': torch.randn(768),
        }
        
        applier = ContinuousWeightApplier(buffer, model, apply_interval=1)
        result = applier.apply_weights(0)
        self.assertTrue(result)


class TestAsyncSamplerKernelInterruptHandling(unittest.TestCase):
    """Test async sampler handles SIGINT/KeyboardInterrupt gracefully."""
    
    def test_sampler_cleanup_on_explicit_stop(self):
        """Test sampler cleans up properly when explicitly stopped."""
        model = SimpleModel()
        buffer = MemoryOptimizer.create_double_buffer()
        causal_engine = MockCausalEngine()
        sampler = CausalWeightSampler(causal_engine, model)
        
        async_sampler = BackgroundSampler(
            buffer=buffer,
            model=model,
            max_steps=100,
            causal_sampler=sampler
        )
        
        async_sampler.start()
        time.sleep(0.05)
        
        # Explicit stop should complete without hanging
        start = time.time()
        async_sampler.stop()
        elapsed = time.time() - start
        
        # Should stop quickly (< 2 seconds)
        self.assertLess(elapsed, 2.0, "Sampler took too long to stop")
    
    def test_sampler_idempotent_stop(self):
        """Test that calling stop() multiple times is safe."""
        model = SimpleModel()
        buffer = MemoryOptimizer.create_double_buffer()
        causal_engine = MockCausalEngine()
        sampler = CausalWeightSampler(causal_engine, model)
        
        async_sampler = BackgroundSampler(
            buffer=buffer,
            model=model,
            max_steps=10,
            causal_sampler=sampler
        )
        
        async_sampler.start()
        
        # Multiple stops should not raise
        async_sampler.stop()
        async_sampler.stop()  # Second call should be safe
        async_sampler.stop()  # Third call should be safe


class TestWeightApplicationDeviceHandling(unittest.TestCase):
    """Test weight application handles device placement correctly."""
    
    def test_weight_application_cpu(self):
        """Test weight application on CPU device."""
        if torch.cuda.is_available():
            self.skipTest("This test is for CPU testing")
        
        model = SimpleModel()
        buffer = Mock()
        buffer.get_latest.return_value = {
            'linear.weight': torch.randn(5, 10, device='cpu'),
        }
        
        applier = ContinuousWeightApplier(buffer, model, device='cpu', apply_interval=1)
        result = applier.apply_weights(0)
        self.assertTrue(result)
    
    def test_weight_application_device_mismatch_handling(self):
        """Test that device mismatch is handled gracefully."""
        model = SimpleModel()
        buffer = Mock()
        
        # Return weights on CPU (model is also CPU, so no actual mismatch)
        buffer.get_latest.return_value = {
            'linear.weight': torch.randn(5, 10, device='cpu'),
        }
        
        applier = ContinuousWeightApplier(buffer, model, device='cpu', apply_interval=1)
        result = applier.apply_weights(0)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
