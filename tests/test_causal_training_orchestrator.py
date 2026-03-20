"""
Integration tests for CausalTrainingOrchestrator.

Tests the complete orchestration of causal weight sampling, application,
and monitoring during training. Uses mocks to isolate orchestrator logic
from actual training/model dependencies.

Test Categories:
1. Initialization - config and component creation
2. Lifecycle - full prepare/train/cleanup cycles
3. Training Integration - weight flow and application timing
4. Diagnostics - metric collection and reporting
5. Error Handling - failure scenarios and recovery
6. Memory - no leaks or errors
"""

import unittest
from unittest.mock import Mock, patch
import torch.nn as nn
from transformers import TrainingArguments, TrainerState

from src.finetuner.causal_training_orchestrator import (
    CausalTrainingOrchestrator,
    WeightApplicationCallback
)
from src.settings.settings import CausalTrainingConfig


class TestCausalTrainingConfig(unittest.TestCase):
    """Tests for CausalTrainingConfig validation and initialization."""
    
    def test_init_with_defaults(self):
        """Test config initializes with reasonable defaults."""
        config = CausalTrainingConfig()
        
        # Core fields
        self.assertEqual(config.total_causal_budget, 1000)
        self.assertEqual(config.async_max_steps, 100)
        self.assertEqual(config.apply_interval, 10)
        self.assertEqual(config.device, "cpu")
        
        # Additional fields
        self.assertEqual(config.seed, 42)
        self.assertIsNone(config.warmup_steps)
        self.assertFalse(config.enable_warmup)
        self.assertEqual(config.logging_level, "INFO")
    
    def test_init_with_custom_values(self):
        """Test config accepts and stores custom values."""
        config = CausalTrainingConfig(
            total_causal_budget=2000,
            async_max_steps=50,
            apply_interval=5,
            device="cuda",
            seed=123,
            warmup_steps=100,
            enable_warmup=True,
            logging_level="DEBUG"
        )
        
        self.assertEqual(config.total_causal_budget, 2000)
        self.assertEqual(config.async_max_steps, 50)
        self.assertEqual(config.apply_interval, 5)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.warmup_steps, 100)
        self.assertTrue(config.enable_warmup)
        self.assertEqual(config.logging_level, "DEBUG")
    
    def test_validation_total_causal_budget_positive(self):
        """Test validation fails if total_causal_budget <= 0."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(total_causal_budget=0)
        
        with self.assertRaises(ValueError):
            CausalTrainingConfig(total_causal_budget=-100)
    
    def test_validation_async_max_steps_positive(self):
        """Test validation fails if async_max_steps <= 0."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(async_max_steps=0)
        
        with self.assertRaises(ValueError):
            CausalTrainingConfig(async_max_steps=-50)
    
    def test_validation_apply_interval_positive(self):
        """Test validation fails if apply_interval <= 0."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(apply_interval=0)
        
        with self.assertRaises(ValueError):
            CausalTrainingConfig(apply_interval=-10)
    
    def test_validation_device_valid(self):
        """Test validation accepts valid device values."""
        for device in ['cpu', 'cuda', 'mps']:
            config = CausalTrainingConfig(device=device)
            self.assertEqual(config.device, device)
    
    def test_validation_device_invalid(self):
        """Test validation rejects invalid device values."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(device='invalid')
    
    def test_validation_device_auto_detection(self):
        """Test 'auto' device is converted to appropriate device."""
        config = CausalTrainingConfig(device='auto')
        # Should be auto-detected to one of: cuda, mps, cpu
        self.assertIn(config.device, ['cuda', 'mps', 'cpu'])
    
    def test_validation_logging_level_case_insensitive(self):
        """Test logging level is case-insensitive and uppercased."""
        config = CausalTrainingConfig(logging_level='info')
        self.assertEqual(config.logging_level, 'INFO')
        
        config = CausalTrainingConfig(logging_level='debug')
        self.assertEqual(config.logging_level, 'DEBUG')
    
    def test_validation_logging_level_valid(self):
        """Test validation accepts valid logging levels."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = CausalTrainingConfig(logging_level=level)
            self.assertEqual(config.logging_level, level)
    
    def test_validation_logging_level_invalid(self):
        """Test validation rejects invalid logging levels."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(logging_level='INVALID')
    
    def test_validation_warmup_steps_positive(self):
        """Test warmu p_steps must be >= 1 if provided."""
        with self.assertRaises(ValueError):
            CausalTrainingConfig(warmup_steps=0)
        
        with self.assertRaises(ValueError):
            CausalTrainingConfig(warmup_steps=-10)
    
    def test_warmup_steps_none_allowed(self):
        """Test warmup_steps can be None."""
        config = CausalTrainingConfig(warmup_steps=None)
        self.assertIsNone(config.warmup_steps)


class TestOrchestratorInitialization(unittest.TestCase):
    """Tests for CausalTrainingOrchestrator initialization."""
    
    def setUp(self):
        """Create mock components for testing."""
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()
    
    def test_init_with_valid_components(self):
        """Test orchestrator initializes with valid components."""
        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
        
        self.assertIsNotNone(orchestrator)
        self.assertEqual(orchestrator.state, orchestrator.IDLE)
    
    def test_init_raises_on_none_lora_engine(self):
        """Test initialization fails if lora_engine is None."""
        with self.assertRaises(ValueError):
            CausalTrainingOrchestrator(
                None,
                self.causal_engine,
                self.trainer,
                self.causal_sampler,
                self.config
            )
    
    def test_init_raises_on_none_causal_engine(self):
        """Test initialization fails if causal_engine is None."""
        with self.assertRaises(ValueError):
            CausalTrainingOrchestrator(
                self.lora_engine,
                None,
                self.trainer,
                self.causal_sampler,
                self.config
            )
    
    def test_init_raises_on_none_trainer(self):
        """Test initialization fails if trainer is None."""
        with self.assertRaises(ValueError):
            CausalTrainingOrchestrator(
                self.lora_engine,
                self.causal_engine,
                None,
                self.causal_sampler,
                self.config
            )
    
    def test_init_raises_on_none_causal_sampler(self):
        """Test initialization fails if causal_sampler is None."""
        with self.assertRaises(ValueError):
            CausalTrainingOrchestrator(
                self.lora_engine,
                self.causal_engine,
                self.trainer,
                None,
                self.config
            )
    
    def test_init_raises_on_none_config(self):
        """Test initialization fails if config is None."""
        with self.assertRaises(ValueError):
            CausalTrainingOrchestrator(
                self.lora_engine,
                self.causal_engine,
                self.trainer,
                self.causal_sampler,
                None
            )


class TestOrchestratorPrepare(unittest.TestCase):
    """Tests for orchestrator.prepare() initialization sequence."""
    
    def setUp(self):
        """Create mock components and orchestrator."""
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()
        
        # Mock model and loader
        self.model = Mock(spec=nn.Module)
        self.data_loader = Mock()
        
        # Mock causal engine methods
        self.causal_engine.identify_causal_paths.return_value = {
            'attention': True,
            'ffn': True
        }
        self.causal_engine.allocate_budget.return_value = None
        self.causal_engine.budget_allocation = {'attention': 500, 'ffn': 500}
        self.causal_engine.get_causal_summary.return_value = {'summary': 'test'}
        
        # Create orchestrator
        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
    
    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_prepare_initializes_all_components(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """Test prepare() initializes all core components in sequence."""
        # Setup mocks
        mock_buffer = Mock()
        mock_optimizer.create_double_buffer.return_value = mock_buffer
        
        mock_sampler_instance = Mock()
        mock_sampler.return_value = mock_sampler_instance
        
        mock_applier_instance = Mock()
        mock_applier.return_value = mock_applier_instance
        
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Call prepare
        self.orchestrator.prepare(self.model, self.data_loader)
        
        # Verify sequence
        self.causal_engine.identify_causal_paths.assert_called_once()
        self.causal_engine.allocate_budget.assert_called_once()
        mock_optimizer.create_double_buffer.assert_called_once()
        mock_sampler.assert_called_once()
        mock_sampler_instance.start.assert_called_once()
        mock_sampler_instance.raise_if_failed.assert_called_once()
        mock_applier.assert_called_once()
        mock_monitor.assert_called_once()
        self.trainer.add_callback.assert_called_once()
    
    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_prepare_updates_state(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """Test prepare() updates state from IDLE to SAMPLING."""
        # Setup mocks
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()
        
        # Initial state should be IDLE
        self.assertEqual(self.orchestrator.state, self.orchestrator.IDLE)
        
        # Call prepare
        self.orchestrator.prepare(self.model, self.data_loader)
        
        # State should now be SAMPLING
        self.assertEqual(self.orchestrator.state, self.orchestrator.SAMPLING)
    
    def test_prepare_raises_if_already_sampling(self):
        """Test prepare() fails if already in SAMPLING state."""
        # Manually set state to SAMPLING (skip actual prepare)
        self.orchestrator._state = self.orchestrator.SAMPLING
        
        # Trying to prepare again should raise ValueError
        with self.assertRaises(ValueError):
            with patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer'):
                self.orchestrator.prepare(self.model, self.data_loader)
    
    def test_prepare_raises_if_already_training(self):
        """Test prepare() fails if already in TRAINING state."""
        # Manually set state to TRAINING
        self.orchestrator._state = self.orchestrator.TRAINING
        
        # Trying to prepare again should raise ValueError
        with self.assertRaises(ValueError):
            with patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer'):
                self.orchestrator.prepare(self.model, self.data_loader)


class TestOrchestratorRunTraining(unittest.TestCase):
    """Tests for orchestrator.run_training() execution."""
    
    def setUp(self):
        """Create orchestrator in SAMPLING state."""
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()
        
        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
        
        # Set up as if prepare() succeeded
        self.orchestrator._state = self.orchestrator.SAMPLING
        self.orchestrator.async_sampler = Mock()
        self.trainer.train.return_value = {'loss': 0.5}
    
    def test_run_training_executes_trainer_train(self):
        """Test run_training() calls trainer.train()."""
        self.orchestrator.run_training()
        
        self.orchestrator.async_sampler.raise_if_failed.assert_called_once()
        self.trainer.train.assert_called_once()
    
    def test_run_training_stops_sampler(self):
        """Test run_training() stops sampler after training."""
        self.orchestrator.run_training()
        
        self.orchestrator.async_sampler.stop.assert_called_once()
    
    def test_run_training_stops_sampler_on_exception(self):
        """Test run_training() stops sampler even if training raises exception."""
        self.trainer.train.side_effect = RuntimeError("Training failed")
        
        with self.assertRaises(RuntimeError):
            self.orchestrator.run_training()
        
        # Sampler should still be stopped
        self.orchestrator.async_sampler.stop.assert_called_once()
    
    def test_run_training_updates_state_to_completed(self):
        """Test run_training() updates state to COMPLETED on success."""
        self.orchestrator.run_training()
        
        self.assertEqual(self.orchestrator.state, self.orchestrator.COMPLETED)
    
    def test_run_training_updates_state_to_failed_on_exception(self):
        """Test run_training() updates state to FAILED on exception."""
        self.trainer.train.side_effect = RuntimeError("Training failed")
        
        with self.assertRaises(RuntimeError):
            self.orchestrator.run_training()
        
        self.assertEqual(self.orchestrator.state, self.orchestrator.FAILED)
    
    def test_run_training_raises_if_not_prepared(self):
        """Test run_training() fails if prepare() not called first."""
        # Create new orchestrator in IDLE state
        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            orchestrator.run_training()


class TestOrchestratorDiagnostics(unittest.TestCase):
    """Tests for orchestrator.get_diagnostics()."""
    
    def setUp(self):
        """Create orchestrator."""
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()
        
        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
    
    def test_diagnostics_dict_keys(self):
        """Test diagnostics returns dict with expected keys."""
        # Set up minimal state
        self.orchestrator._state = self.orchestrator.COMPLETED
        self.orchestrator.causal_engine.get_causal_summary.return_value = {}
        mock_monitor = Mock()
        mock_monitor.get_budget_utilization.return_value = {}
        self.orchestrator.budget_monitor = mock_monitor
        self.trainer.state = Mock(best_metric=0.9)
        
        diagnostics = self.orchestrator.get_diagnostics()
        
        # Check all expected keys present
        self.assertIn('state', diagnostics)
        self.assertIn('causal_summary', diagnostics)
        self.assertIn('budget_utilization', diagnostics)
        self.assertIn('training_metrics', diagnostics)
        self.assertIn('config', diagnostics)
    
    def test_diagnostics_config_values(self):
        """Test diagnostics includes correct config values."""
        self.orchestrator._state = self.orchestrator.COMPLETED
        
        diagnostics = self.orchestrator.get_diagnostics()
        config_dict = diagnostics['config']
        
        self.assertEqual(config_dict['total_causal_budget'], 1000)
        self.assertEqual(config_dict['async_max_steps'], 100)
        self.assertEqual(config_dict['apply_interval'], 10)
        self.assertEqual(config_dict['device'], 'cpu')
        self.assertEqual(config_dict['seed'], 42)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and graceful degradation."""
    
    def setUp(self):
        """Create orchestrator components."""
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()
        
        self.model = Mock(spec=nn.Module)
        self.data_loader = Mock()
        
        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config
        )
    
    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_prepare_failure_sets_failed_state(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """Test prepare() sets FAILED state if exception occurs."""
        # Make identify_causal_paths fail
        self.causal_engine.identify_causal_paths.side_effect = RuntimeError("Path identification failed")
        
        with self.assertRaises(RuntimeError):
            self.orchestrator.prepare(self.model, self.data_loader)
        
        self.assertEqual(self.orchestrator.state, self.orchestrator.FAILED)
    
    def test_weight_callback_handles_exceptions(self):
        """Test WeightApplicationCallback handles applier exceptions gracefully."""
        applier = Mock()
        applier.apply_weights.side_effect = RuntimeError("Apply failed")
        
        callback = WeightApplicationCallback(applier)
        
        # Create mock trainer state
        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        state.global_step = 10
        control = Mock()
        
        # Should not raise (handles exception internally)
        result = callback.on_step_end(args, state, control)
        
        # Should return control unchanged
        self.assertEqual(result, control)

    def test_weight_callback_captures_sampler_health_errors(self):
        """Test callback logs and stores background sampler failures."""
        applier = Mock()
        sampler_health_check = Mock(side_effect=RuntimeError("Worker failed"))

        callback = WeightApplicationCallback(
            applier,
            sampler_health_check=sampler_health_check,
        )

        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        state.global_step = 10
        control = Mock()

        result = callback.on_step_end(args, state, control)

        self.assertEqual(result, control)
        sampler_health_check.assert_called_once()
        applier.apply_weights.assert_not_called()
        self.assertEqual(callback.last_error, "Worker failed")


class TestMemory(unittest.TestCase):
    """Memory and lifecycle tests."""
    
    def test_reset_clears_components(self):
        """Test reset() clears all components."""
        config = CausalTrainingConfig()
        orchestrator = CausalTrainingOrchestrator(
            Mock(), Mock(), Mock(), Mock(), config
        )
        
        # Set up components
        orchestrator.buffer = Mock()
        orchestrator.async_sampler = Mock()
        orchestrator.weight_applier = Mock()
        orchestrator.budget_monitor = Mock()
        orchestrator.weight_callback = Mock()
        
        # Call reset
        orchestrator.reset()
        
        # All components should be None
        self.assertIsNone(orchestrator.buffer)
        self.assertIsNone(orchestrator.async_sampler)
        self.assertIsNone(orchestrator.weight_applier)
        self.assertIsNone(orchestrator.budget_monitor)
        self.assertIsNone(orchestrator.weight_callback)
        self.assertEqual(orchestrator.state, orchestrator.IDLE)


if __name__ == '__main__':
    unittest.main()
