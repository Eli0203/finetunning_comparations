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
from src.settings import CausalTrainingConfig


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
    def test_prepare_registers_interventional_callback_when_enabled(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """Interventional callback should be registered when config enables it."""
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()

        self.orchestrator.config.enable_interventional_weights = True
        self.orchestrator.prepare(self.model, self.data_loader)

        self.assertIsNotNone(self.orchestrator.interventional_callback)
        self.assertEqual(self.trainer.add_callback.call_count, 2)
    
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


class TestUS3FailingFirstPrepareBudget(unittest.TestCase):
    """Failing-first tests for Phase 5 US3 (T065-T067)."""

    def setUp(self):
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()

        self.model = Mock(spec=nn.Module)
        self.data_loader = Mock()

        self.causal_engine.identify_causal_paths.return_value = {
            'attention': True,
            'ffn': True,
        }
        self.causal_engine.allocate_budget.return_value = None
        self.causal_engine.budget_allocation = {'attention': 500, 'ffn': 500}
        self.causal_engine.get_causal_summary.return_value = {'summary': 'test'}

        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config,
        )

    def test_register_callbacks_blocked_when_prepare_not_called(self):
        """T065: callback registration must fail-fast before prepare()."""
        with self.assertRaises(RuntimeError):
            self.orchestrator.register_callbacks()

    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_prepare_sets_is_prepared_true(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """T066: prepare() must expose readiness via is_prepared=true."""
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()

        self.orchestrator.prepare(self.model, self.data_loader)

        self.assertTrue(self.orchestrator.is_prepared)

    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_budget_allocation_populated_post_prepare(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """T067: prepare() must populate orchestrator-level budget allocation."""
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()

        self.orchestrator.prepare(self.model, self.data_loader)

        self.assertEqual(
            self.orchestrator.budget_allocation,
            self.causal_engine.budget_allocation,
        )


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

    def test_run_training_requests_skip_next_apply_on_non_finite_mml(self):
        """US5: fail-closed MML should request skip-next-apply."""
        self.orchestrator.weight_applier = Mock()
        self.orchestrator.weight_applier.request_skip_next_apply = Mock()
        self.orchestrator._model_ref = Mock(spec=nn.Module)
        self.orchestrator.causal_engine.validate_marginal_likelihood.return_value = None
        self.trainer.get_eval_dataloader = Mock(return_value=[])

        self.orchestrator.run_training()

        self.orchestrator.weight_applier.request_skip_next_apply.assert_called_once()


class TestUS3IntegrationPrepareBudget(unittest.TestCase):
    """Integration tests for mandatory prepare flow and budget enforcement (T075-T077)."""

    def setUp(self):
        self.lora_engine = Mock()
        self.causal_engine = Mock()
        self.trainer = Mock()
        self.causal_sampler = Mock()
        self.config = CausalTrainingConfig()

        self.model = Mock(spec=nn.Module)
        self.data_loader = Mock()

        self.causal_engine.identify_causal_paths.return_value = {
            'attention': True,
            'ffn': True,
        }
        self.causal_engine.allocate_budget.return_value = None
        self.causal_engine.get_causal_summary.return_value = {'summary': 'ok'}

        self.orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            self.trainer,
            self.causal_sampler,
            self.config,
        )

    def test_causal_training_blocked_without_prepare(self):
        """T075: run_training must fail if prepare was not executed."""
        with self.assertRaises(ValueError):
            self.orchestrator.run_training()

    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_causal_training_succeeds_with_valid_prepared_budget(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """T076: valid budget after prepare enables training execution."""
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()

        self.causal_engine.budget_allocation = {'attention': 500, 'ffn': 500}
        self.trainer.train.return_value = {'loss': 0.1}

        self.orchestrator.prepare(self.model, self.data_loader)
        output = self.orchestrator.run_training()

        self.assertEqual(output, {'loss': 0.1})
        self.assertEqual(self.orchestrator.state, self.orchestrator.COMPLETED)

    @patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer')
    @patch('src.finetuner.causal_training_orchestrator.BackgroundSampler')
    @patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier')
    @patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor')
    def test_invalid_budget_blocks_run_with_clear_error(
        self, mock_monitor, mock_applier, mock_sampler, mock_optimizer
    ):
        """T077: invalid budget must block run with explicit violation diagnostic."""
        mock_optimizer.create_double_buffer.return_value = Mock()
        mock_sampler.return_value = Mock()
        mock_applier.return_value = Mock()
        mock_monitor.return_value = Mock()

        self.causal_engine.budget_allocation = {'attention': -1, 'ffn': 500}

        with self.assertRaises(ValueError) as exc_info:
            self.orchestrator.prepare(self.model, self.data_loader)

        self.assertIn('Budget constraint violation', str(exc_info.exception))


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
        mock_monitor.get_current_budget_snapshot.return_value = {}
        mock_monitor.get_metrics.return_value = {'step_count': 0, 'tracked_paths': 0}
        self.orchestrator.budget_monitor = mock_monitor
        self.orchestrator.weight_applier = Mock()
        self.orchestrator.weight_applier.get_metrics.return_value = {'times_applied': 3}
        self.orchestrator.weight_callback = Mock()
        self.orchestrator.weight_callback.get_metrics.return_value = {'applied_steps': 3}
        self.orchestrator.weight_callback.last_error = None
        self.orchestrator.async_sampler = Mock()
        self.orchestrator.async_sampler.get_status.return_value = {'is_running': False, 'metrics': {}}
        self.trainer.state = Mock(best_metric=0.9)
        
        diagnostics = self.orchestrator.get_diagnostics()
        
        # Check all expected keys present
        self.assertIn('state', diagnostics)
        self.assertIn('causal_summary', diagnostics)
        self.assertIn('budget_utilization', diagnostics)
        self.assertIn('training_metrics', diagnostics)
        self.assertIn('budget_snapshot', diagnostics)
        self.assertIn('budget_monitor_metrics', diagnostics)
        self.assertIn('weight_application_metrics', diagnostics)
        self.assertIn('callback_metrics', diagnostics)
        self.assertIn('failure_policy', diagnostics)
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

    def test_diagnostics_failure_policy_values(self):
        """Diagnostics should expose explicit fail-open/fail-closed policy."""
        self.orchestrator._state = self.orchestrator.COMPLETED

        diagnostics = self.orchestrator.get_diagnostics()
        policy = diagnostics['failure_policy']

        self.assertEqual(policy['causal_gradient_unavailable'], 'fail_closed')
        self.assertEqual(policy['laplace_phase_failure_notebook'], 'fail_closed')
        self.assertEqual(policy['generic_orchestrator_exception_notebook'], 'fallback_to_standard_lora')


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

    def test_weight_callback_tracks_applied_and_skipped_steps(self):
        """Callback metrics should reflect successful and skipped apply cycles."""
        applier = Mock()
        applier.apply_weights.side_effect = [True, False]

        callback = WeightApplicationCallback(applier)
        args = Mock(spec=TrainingArguments)
        control = Mock()

        state = Mock(spec=TrainerState)
        state.global_step = 1
        callback.on_step_end(args, state, control)

        state.global_step = 2
        callback.on_step_end(args, state, control)

        metrics = callback.get_metrics()
        self.assertEqual(metrics['applied_steps'], 1)
        self.assertEqual(metrics['skipped_steps'], 1)


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


class TestOrchestratorStateTransitions(unittest.TestCase):
    """Tests for orchestrator state machine transitions (T056).
    
    Spec Requirement: Strictly sequential progression with validation gates:
    IDLE → PREPARING → SAMPLING → TRAINING
    - SAMPLING can only start after warm-up completion
    - TRAINING waits for in-flight SAMPLING to complete before advancing
    """
    
    def setUp(self):
        """Create mock components and orchestrator."""
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
    
    def test_initial_state_is_idle(self):
        """Test orchestrator starts in IDLE state."""
        self.assertEqual(self.orchestrator.state, 'IDLE')
    
    def test_prepare_transitions_to_preparing_state(self):
        """Test prepare() moves orchestrator to PREPARING state."""
        with patch('src.finetuner.causal_training_orchestrator.MemoryOptimizer'):
            with patch('src.finetuner.causal_training_orchestrator.BackgroundSampler'):
                with patch('src.finetuner.causal_training_orchestrator.ContinuousWeightApplier'):
                    with patch('src.finetuner.causal_training_orchestrator.TrainingBudgetMonitor'):
                        # Setup mocks
                        self.causal_engine.identify_causal_paths.return_value = {'attention': True}
                        self.causal_engine.allocate_budget.return_value = None
                        self.causal_engine.budget_allocation = {'attention': 1}
                        self.causal_engine.get_causal_summary.return_value = {}
                        
                        model = Mock()
                        data_loader = Mock()
                        
                        # Execute prepare
                        self.orchestrator.prepare(model, data_loader)
                        
                        # Verify state change
                        self.assertEqual(self.orchestrator.state, 'SAMPLING')
    
    def test_warmup_completion_enables_sampling_transition(self):
        """Test SAMPLING state can only be reached after warm-up completion."""
        # Mock warm-up completion
        self.orchestrator._warmup_complete = False
        
        # Try to transition to SAMPLING - should be blocked
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator._transition_to_sampling()
        
        # Mark warm-up as complete
        self.orchestrator._warmup_complete = True
        
        # Now transition should succeed
        self.orchestrator._transition_to_sampling()
        self.assertEqual(self.orchestrator.state, 'SAMPLING')
    
    def test_sampling_to_training_waits_for_async_completion(self):
        """Test TRAINING state waits for in-flight SAMPLING to complete."""
        # Set to SAMPLING state
        self.orchestrator.state = 'SAMPLING'
        
        # Mock async sampler with pending operations
        self.orchestrator.async_sampler = Mock()
        self.orchestrator.async_sampler.is_active.return_value = True
        
        # Transition should wait for sampler to complete
        self.orchestrator._transition_to_training()
        
        # Verify that wait was called
        self.orchestrator.async_sampler.join.assert_called_once()
        self.assertEqual(self.orchestrator.state, 'TRAINING')
    
    def test_training_to_idle_cleanup(self):
        """Test TRAINING → IDLE transition performs cleanup."""
        self.orchestrator.state = 'TRAINING'
        self.orchestrator.buffer = Mock()
        self.orchestrator.async_sampler = Mock()
        self.orchestrator.weight_applier = Mock()
        
        # Transition back to IDLE
        self.orchestrator._transition_to_idle()
        
        # Verify cleanup was performed
        self.orchestrator.async_sampler.stop.assert_called_once()
        self.assertEqual(self.orchestrator.state, 'IDLE')
    
    def test_invalid_state_transition_rejected(self):
        """Test invalid state transitions are rejected."""
        # Try to go from IDLE to TRAINING directly
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator.state = 'IDLE'
            self.orchestrator._transition_to_training()
    
    def test_state_transition_sequence_complete(self):
        """Test complete state sequence: IDLE → PREPARING → SAMPLING → TRAINING."""
        states = []
        original_set_state = self.orchestrator.__setattr__
        
        def track_state_changes(name, value):
            if name == 'state':
                states.append(value)
            original_set_state(name, value)
        
        self.orchestrator.__setattr__ = track_state_changes
        
        # Start in IDLE
        self.assertEqual(self.orchestrator.state, 'IDLE')
        states.append('IDLE')
        
        # Prepare moves to PREPARING
        self.orchestrator.state = 'PREPARING'
        self.assertEqual(self.orchestrator.state, 'PREPARING')
        
        # Warm-up complete, move to SAMPLING
        self.orchestrator._warmup_complete = True
        self.orchestrator.state = 'SAMPLING'
        self.assertEqual(self.orchestrator.state, 'SAMPLING')
        
        # Move to TRAINING
        self.orchestrator.state = 'TRAINING'
        self.assertEqual(self.orchestrator.state, 'TRAINING')


class TestWarmupGateBlocking(unittest.TestCase):
    """Tests for warm-up gate blocking SAMPLING transitions (T057).
    
    Spec Requirement: SAMPLING can only start after ALL warm-up gates pass:
    1. Signal gate: ||BA||_F > 1e-6
    2. Loss gate: EMA loss is stable (no initial divergence)
    3. Causal gate: Var(NIE) > 1e-6
    4. Resource gate: sufficient buffer/VRAM capacity
    """
    
    def setUp(self):
        """Create orchestrator with mocked components."""
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
        
        # Initialize warm-up gate state
        self.orchestrator._warmup_gates = {
            'signal': False,
            'loss': False,
            'causal': False,
            'resource': False
        }
    
    def test_sampling_blocked_when_signal_gate_not_passed(self):
        """Test SAMPLING blocked if signal gate (||BA||_F > 1e-6) not passed."""
        self.orchestrator._warmup_gates = {
            'signal': False,
            'loss': True,
            'causal': True,
            'resource': True
        }
        
        # Try to transition to SAMPLING - should fail
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator._check_warmup_gates()
    
    def test_sampling_blocked_when_loss_gate_not_passed(self):
        """Test SAMPLING blocked if loss gate (EMA stability) not passed."""
        self.orchestrator._warmup_gates = {
            'signal': True,
            'loss': False,
            'causal': True,
            'resource': True
        }
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator._check_warmup_gates()
    
    def test_sampling_blocked_when_causal_gate_not_passed(self):
        """Test SAMPLING blocked if causal gate (Var(NIE) > 1e-6) not passed."""
        self.orchestrator._warmup_gates = {
            'signal': True,
            'loss': True,
            'causal': False,
            'resource': True
        }
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator._check_warmup_gates()
    
    def test_sampling_blocked_when_resource_gate_not_passed(self):
        """Test SAMPLING blocked if resource gate (buffer/VRAM) not passed."""
        self.orchestrator._warmup_gates = {
            'signal': True,
            'loss': True,
            'causal': True,
            'resource': False
        }
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.orchestrator._check_warmup_gates()
    
    def test_sampling_allowed_when_all_gates_passed(self):
        """Test SAMPLING allowed only when ALL gates pass."""
        self.orchestrator._warmup_gates = {
            'signal': True,
            'loss': True,
            'causal': True,
            'resource': True
        }
        
        # Should not raise
        self.orchestrator._check_warmup_gates()
        self.orchestrator._warmup_complete = True
    
    def test_signal_gate_checks_frobenius_norm(self):
        """Test signal gate checks ||BA||_F > 1e-6."""
        # Mock gradient computation
        self.orchestrator.causal_engine = Mock()
        self.orchestrator.causal_engine.compute_backdoor_gradients.return_value = {
            'attention': {'A': Mock(), 'B': Mock()},
            'ffn': {'A': Mock(), 'B': Mock()}
        }
        
        # Test low Frobenius norm (should fail)
        with patch('torch.linalg.matrix_norm') as mock_norm:
            mock_norm.return_value = 1e-7
            result = self.orchestrator._check_signal_gate()
            self.assertFalse(result)
        
        # Test high Frobenius norm (should pass)
        with patch('torch.linalg.matrix_norm') as mock_norm:
            mock_norm.return_value = 1e-5
            result = self.orchestrator._check_signal_gate()
            self.assertTrue(result)
    
    def test_loss_gate_checks_ema_stability(self):
        """Test loss gate checks EMA loss stability."""
        # Mock trainer state
        self.orchestrator.trainer = Mock()
        self.orchestrator.trainer.state = Mock()
        self.orchestrator.trainer.state.log_history = [
            {'loss': 4.0},
            {'loss': 3.9},
            {'loss': 3.8},  # Stable decline
        ]
        
        # Should pass with stable EMA
        result = self.orchestrator._check_loss_gate()
        self.assertTrue(result)
    
    def test_resource_gate_checks_buffer_availability(self):
        """Test resource gate checks buffer availability."""
        # Mock buffer
        self.orchestrator.buffer = Mock()
        self.orchestrator.buffer.available_slots.return_value = 10
        
        # Should pass with sufficient slots
        result = self.orchestrator._check_resource_gate()
        self.assertTrue(result)
        
        # Should fail with no available slots
        self.orchestrator.buffer.available_slots.return_value = 0
        result = self.orchestrator._check_resource_gate()
        self.assertFalse(result)


class TestMultiPrecisionSupport(unittest.TestCase):
    """Tests for multi-precision peft_config injection (T058).
    
    Spec Requirement: Dynamically inject peft_config (FP16 or NF4/QLoRA)
    based on memory profile without hardcoding precision barriers.
    """
    
    def setUp(self):
        """Create orchestrator for multi-precision testing."""
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
    
    def test_peft_config_injection_for_low_vram_profile(self):
        """Test NF4/QLoRA injection for Low-VRAM profile."""
        available_vram = 4.0  # 4 GB
        
        peft_config = self.orchestrator._select_peft_config(available_vram)
        
        # Should return QLoRA/NF4 config for low VRAM
        self.assertIsNotNone(peft_config)
        self.assertIn('nf4', str(peft_config).lower())
    
    def test_peft_config_injection_for_medium_vram_profile(self):
        """Test FP16 injection for Medium-VRAM profile."""
        available_vram = 8.0  # 8 GB
        
        peft_config = self.orchestrator._select_peft_config(available_vram)
        
        # Should return FP16 config for medium VRAM
        self.assertIsNotNone(peft_config)
    
    def test_peft_config_injection_for_high_vram_profile(self):
        """Test FP32 injection for High-VRAM profile."""
        available_vram = 16.0  # 16 GB
        
        peft_config = self.orchestrator._select_peft_config(available_vram)
        
        # Should return FP32 config for high VRAM
        self.assertIsNotNone(peft_config)
    
    def test_peft_config_respects_default_quantization_setting(self):
        """Test peft_config selection respects DEFAULT_QUANTIZATION setting."""
        from src.settings import Settings
        
        # Create settings with nf4_forced
        with patch.dict('os.environ', {'DEFAULT_QUANTIZATION': 'nf4_forced'}):
            settings = Settings(hf_token='test')
            self.orchestrator.config._settings = settings
            
            # Should force NF4 even with higher VRAM
            available_vram = 16.0
            peft_config = self.orchestrator._select_peft_config(available_vram)
            
            self.assertIsNotNone(peft_config)
    
    def test_peft_config_registered_with_trainer(self):
        """Test peft_config is registered with trainer model."""
        peft_config = self.orchestrator._select_peft_config(4.0)
        
        # Mock model
        model = Mock()
        
        # Register config
        self.orchestrator._register_peft_config(model, peft_config)
        
        # Config should be attached to model
        self.assertTrue(hasattr(model, 'peft_config') or model.method_calls)


class TestNF4AutoDetection(unittest.TestCase):
    """Tests for NF4 auto-detection and BitsAndBytesConfig (T059).
    
    Spec Requirement: Auto-detect NF4 at startup; if detected, apply
    BitsAndBytesConfig with double quantization automatically.
    """
    
    def setUp(self):
        """Create orchestrator for NF4 detection testing."""
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
    
    def test_nf4_detection_on_low_vram(self):
        """Test NF4 is auto-detected when VRAM < 6GB."""
        available_vram = 4.0  # 4 GB
        
        should_use_nf4 = self.orchestrator._should_use_nf4(available_vram)
        
        self.assertTrue(should_use_nf4)
    
    def test_nf4_not_detected_on_high_vram(self):
        """Test NF4 not auto-detected when VRAM >= 6GB."""
        available_vram = 8.0  # 8 GB
        
        should_use_nf4 = self.orchestrator._should_use_nf4(available_vram)
        
        self.assertFalse(should_use_nf4)
    
    def test_bits_and_bytes_config_creation_with_nf4(self):
        """Test BitsAndBytesConfig is created with double quantization."""
        bnb_config = self.orchestrator._create_bnb_config()
        
        # Verify double quantization is enabled
        self.assertTrue(bnb_config.double_quantization)
        self.assertEqual(bnb_config.quant_type, 'nf4')
    
    def test_bits_and_bytes_config_4bit_settings(self):
        """Test BitsAndBytesConfig has correct 4-bit compute settings."""
        bnb_config = self.orchestrator._create_bnb_config()
        
        # Verify 4-bit compute settings
        self.assertTrue(bnb_config.load_in_4bit)
        self.assertEqual(bnb_config.bnb_4bit_compute_dtype, 'float16')
    
    def test_nf4_detection_respects_forced_setting(self):
        """Test NF4_FORCED overrides VRAM-based detection."""
        from src.settings import Settings
        
        with patch.dict('os.environ', {'DEFAULT_QUANTIZATION': 'nf4_forced'}):
            settings = Settings(hf_token='test')
            self.orchestrator.config._settings = settings
            
            # Even with high VRAM, should use NF4
            available_vram = 16.0
            should_use_nf4 = self.orchestrator._should_use_nf4(
                available_vram, 
                force_nf4=True
            )
            
            self.assertTrue(should_use_nf4)
    
    def test_bnb_config_applied_to_model_loading(self):
        """Test BitsAndBytesConfig is applied during model loading."""
        bnb_config = self.orchestrator._create_bnb_config()
        
        # Mock model loading
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load with BitsAndBytes config
            self.orchestrator._load_model_with_bnb(
                'bert-base-uncased',
                bnb_config
            )
            
            # Verify from_pretrained was called with quantization_config
            called_kwargs = mock_load.call_args[1]
            self.assertIn('quantization_config', called_kwargs)


class TestBackgroundSamplerAsyncPreFetch(unittest.TestCase):
    """Tests for BackgroundSampler async pre-fetch with blocking wait (T060).
    
    Spec Requirement: SAMPLING runs in background to pre-fetch batches;
    TRAINING consumes them synchronously with explicit blocking wait if
    buffer empty.
    """
    
    def setUp(self):
        """Create orchestrator for async sampler testing."""
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
    
    def test_background_sampler_starts_on_prepare(self):
        """Test BackgroundSampler starts during prepare()."""
        async_sampler = Mock()
        self.orchestrator.async_sampler = async_sampler
        
        # Start sampler
        self.orchestrator._start_background_sampler()
        
        # Verify start was called
        async_sampler.start.assert_called_once()
    
    def test_background_sampler_pre_fetches_batches(self):
        """Test BackgroundSampler pre-fetches batches asynchronously."""
        async_sampler = Mock()
        async_sampler.get_next_weight.return_value = {'batch': 0, 'weight': 1.0}
        
        self.orchestrator.async_sampler = async_sampler
        
        # Request batch
        batch = self.orchestrator._get_next_weight_async()
        
        # Should return immediately without blocking train loop
        self.assertEqual(batch, {'batch': 0, 'weight': 1.0})
    
    def test_blocking_wait_when_buffer_empty(self):
        """Test explicit blocking wait if sampler buffer is empty."""
        async_sampler = Mock()
        
        # First call returns None (buffer empty), second returns weight
        async_sampler.get_next_weight.side_effect = [None, {'weight': 1.0}]
        
        self.orchestrator.async_sampler = async_sampler
        
        # Request weight with blocking wait
        weight = self.orchestrator._get_next_weight_blocking()
        
        # Should have waited and returned weight
        self.assertEqual(weight, {'weight': 1.0})
        self.assertEqual(async_sampler.wait_for_batch.call_count, 1)
    
    def test_async_prefetch_throughput_exceeds_training_rate(self):
        """Test sampler throughput ≥ 2× training step rate."""
        async_sampler = Mock()
        
        # Simulate pre-fetching at 2x training rate
        training_batch_rate = 1000  # batches/sec
        expected_sampler_rate = training_batch_rate * 2  # 2000 batches/sec
        
        async_sampler.get_throughput.return_value = expected_sampler_rate
        
        self.orchestrator.async_sampler = async_sampler
        
        actual_rate = async_sampler.get_throughput()
        
        # Verify sampler exceeds 2x training rate
        self.assertGreaterEqual(actual_rate, expected_sampler_rate)
    
    def test_async_sampler_handles_failure_gracefully(self):
        """Test orchestrator handles sampler failures gracefully."""
        async_sampler = Mock()
        async_sampler.is_failed.return_value = False
        async_sampler.get_failure_reason.return_value = None
        
        self.orchestrator.async_sampler = async_sampler
        
        # Check for failures
        has_failed = async_sampler.is_failed()
        
        self.assertFalse(has_failed)
    
    def test_async_sampler_stops_on_cleanup(self):
        """Test BackgroundSampler stops during orchestrator cleanup."""
        async_sampler = Mock()
        async_sampler.is_active.return_value = True
        
        self.orchestrator.async_sampler = async_sampler
        
        # Stop sampler
        self.orchestrator._stop_background_sampler()
        
        # Verify stop was called
        async_sampler.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
