"""
Causal Training Orchestrator for coordinating end-to-end causal weight sampling and application.

This module orchestrates the complete causal training pipeline:
1. Identifies causal paths in the model
2. Allocates budget based on causal importance
3. Asynchronously samples weights informed by causal budget
4. Continuously applies weights during training at regular intervals
5. Monitors budget consumption and provides diagnostics

Single Responsibility: This class coordinates all components but implements
NO core logic itself. All actual work is delegated to specialized components.
"""

import torch.nn as nn
from typing import Any, Callable, Dict, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from src.utils.logger import logger
from src.utils.causal_sampler import CausalWeightSampler
from src.utils.async_sampler import BackgroundSampler
from src.utils.training_integrator import ContinuousWeightApplier, TrainingBudgetMonitor
from src.utils.memory_manager import MemoryOptimizer
from src.settings.settings import CausalTrainingConfig


class WeightApplicationCallback(TrainerCallback):
    """
    Custom HuggingFace Trainer callback for applying causal weights at each step.
    
    This callback integrates with the Trainer's callback system to apply
    weights at the configured interval during training.
    """
    
    def __init__(
        self,
        weight_applier: ContinuousWeightApplier,
        sampler_health_check: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the callback.
        
        Args:
            weight_applier: ContinuousWeightApplier instance for applying weights
            sampler_health_check: Optional callable that raises if the background
                sampler has failed in another process.
        """
        self.weight_applier = weight_applier
        self.sampler_health_check = sampler_health_check
        self.step_count = 0
        self.last_error: Optional[str] = None
        self.applied_steps = 0
        self.skipped_steps = 0
        self.health_check_failures = 0
        self.application_failures = 0
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """
        Called at the end of each training step.
        
        Attempts to apply weights if interval reached. Gracefully handles
        any failures without stopping training.
        """
        try:
            if self.sampler_health_check is not None:
                self.sampler_health_check()
            global_step = state.global_step
            weights_applied = self.weight_applier.apply_weights(global_step)
            if weights_applied:
                self.applied_steps += 1
                logger.debug(f"Applied causal weights at step {global_step}")
            else:
                self.skipped_steps += 1
        except Exception as e:
            self.last_error = str(e)
            self.health_check_failures += 1
            self.application_failures += 1
            logger.error(
                f"Error applying weights at step {state.global_step}: {e}. "
                "Continuing training without weight application."
            )
        
        return control

    def get_metrics(self) -> Dict[str, Any]:
        """Return callback-level application and failure counters."""
        return {
            'applied_steps': self.applied_steps,
            'skipped_steps': self.skipped_steps,
            'health_check_failures': self.health_check_failures,
            'application_failures': self.application_failures,
            'last_error': self.last_error,
        }


class CausalTrainingOrchestrator:
    """
    Orchestrator for end-to-end causal training with continuous weight application.
    
    Coordinates the complete causal training pipeline:
    - Causal path identification and budget allocation
    - Async weight sampling informed by causal budget
    - Continuous weight application during training
    - Budget consumption monitoring and diagnostics
    
    Single Responsibility: Orchestrate component interactions, do NOT implement
    core logic. All actual work delegated to specialized components.
    
    Design Patterns:
    - Dependency Injection: All components provided at construction
    - State Machine: Tracks orchestrator state (IDLE → PREPARING → SAMPLING → TRAINING)
    - Template Method: prepare() initializes in correct sequence
    - Composition: Uses specialized components rather than inheritance
    """
    
    # State constants
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    SAMPLING = "SAMPLING"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    
    def __init__(
        self,
        lora_engine: Any,  # FineTuningEngine
        causal_engine: Any,  # CausalMonteCLoRAEngine
        trainer: Any,  # HuggingFace Trainer
        causal_sampler: CausalWeightSampler,
        config: CausalTrainingConfig,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            lora_engine: FineTuningEngine for LoRA adaptation
            causal_engine: CausalMonteCLoRAEngine for causal analysis
            trainer: HuggingFace Trainer instance
            causal_sampler: CausalWeightSampler for intelligent sampling
            config: CausalTrainingConfig with orchestration parameters
            
        Raises:
            ValueError: If any required component is None
        """
        if not all([lora_engine, causal_engine, trainer, causal_sampler, config]):
            raise ValueError("All orchestrator components must be provided (not None)")
        
        self.lora_engine = lora_engine
        self.causal_engine = causal_engine
        self.trainer = trainer
        self.causal_sampler = causal_sampler
        self.config = config
        
        # Components created in prepare()
        self.buffer: Optional[Any] = None
        self.async_sampler: Optional[BackgroundSampler] = None
        self.weight_applier: Optional[ContinuousWeightApplier] = None
        self.budget_monitor: Optional[TrainingBudgetMonitor] = None
        self.weight_callback: Optional[WeightApplicationCallback] = None
        self._model_ref: Optional[nn.Module] = None
        self._train_loader_ref: Optional[Any] = None
        
        # State tracking
        self._state = self.IDLE
        logger.info("CausalTrainingOrchestrator initialized in IDLE state")
    
    @property
    def state(self) -> str:
        """Return current orchestrator state."""
        return self._state
    
    def prepare(self, model: nn.Module, data_loader: Any) -> None:
        """
        Initialize all components for causal training.
        
        Orchestrates creation and startup of:
        1. Causal path identification
        2. Budget allocation
        3. Double buffer for IPC
        4. Background sampler
        5. Weight applier
        6. Budget monitor
        7. Trainer callback for weight application
        
        Args:
            model: PyTorch model with LoRA adapters
            data_loader: Training data loader
            
        Raises:
            ValueError: If already in SAMPLING or TRAINING state
            Exception: If any component initialization fails
        """
        if self._state in [self.SAMPLING, self.TRAINING]:
            raise ValueError(
                f"Cannot call prepare() when in {self._state} state. "
                "Call reset() first or use new orchestrator instance."
            )
        
        try:
            logger.info("=" * 60)
            logger.info("PHASE 6: Preparing Causal Training Orchestrator...")
            logger.info("=" * 60)
            
            self._state = self.PREPARING
            self._model_ref = model
            self._train_loader_ref = data_loader
            
            # Step 1: Identify causal paths
            logger.info("[1/7] Identifying causal paths in model (using backdoor adjustment)...")
            causal_paths = self.causal_engine.identify_causal_paths(
                model, data_loader
            )
            logger.info(f"[1/7] ✓ Identified {len(causal_paths)} causal paths")
            # Log first few paths for diagnostics (handle dict/list returns from mocks)
            paths_to_log = list(causal_paths) if isinstance(causal_paths, (list, tuple)) else list(causal_paths.keys()) if isinstance(causal_paths, dict) else causal_paths
            for i, path in enumerate(paths_to_log[:5], 1):
                logger.debug(f"      Path {i}: {path}")
            if len(paths_to_log) > 5:
                logger.debug(f"      ... and {len(paths_to_log) - 5} more")
            
            # Step 2: Allocate budget
            logger.info("[2/7] Allocating causal budget (mediation analysis)...")
            self.causal_engine.allocate_budget(
                causal_paths,
                self.config.total_causal_budget
            )
            budget_summary = self.causal_engine.budget_allocation
            logger.info(f"[2/7] ✓ Allocated budget across {len(budget_summary)} paths")
            for path, budget in list(budget_summary.items())[:3]:
                logger.debug(f"      {path}: {budget} samples")

            # Notify the sampler so it can re-read the now-populated budget.
            # This eliminates the "no budget allocation" warning that fires when
            # the sampler is constructed before prepare() is called.
            self.causal_sampler.refresh_path_weights()
            logger.info("[2/7] ✓ Sampler path weights refreshed from causal budget")
            
            # Step 3: Create double buffer for inter-process communication
            logger.info("[3/7] Creating double buffer for weight communication (O(1) retrieval)...")
            self.buffer = MemoryOptimizer.create_double_buffer()
            logger.info("[3/7] ✓ Double buffer created")
            
            # Step 4: Create and start background sampler
            logger.info("[4/7] Starting background weight sampler (async multiprocessing)...")
            self.async_sampler = BackgroundSampler(
                buffer=self.buffer,
                model=model,
                max_steps=self.config.async_max_steps,
                causal_sampler=self.causal_sampler
            )
            self.async_sampler.start()
            self.async_sampler.raise_if_failed()
            logger.info(f"[4/7] ✓ Background sampler started ({self.config.async_max_steps} max steps)")
            
            # Step 5: Create weight applier
            logger.info("[5/7] Creating continuous weight applier (rate-limited application)...")
            self.weight_applier = ContinuousWeightApplier(
                buffer=self.buffer,
                model=model,
                device=self.config.device,
                apply_interval=self.config.apply_interval
            )
            logger.info(f"[5/7] ✓ Weight applier configured (interval: {self.config.apply_interval} steps)")
            
            # Step 6: Create budget monitor
            logger.info("[6/7] Creating training budget monitor (utilization tracking)...")
            self.budget_monitor = TrainingBudgetMonitor(self.causal_engine)
            logger.info("[6/7] ✓ Budget monitor initialized")
            
            # Step 7: Register trainer callback for weight application
            logger.info("[7/7] Registering weight application callback with HuggingFace Trainer...")
            self.weight_callback = WeightApplicationCallback(
                self.weight_applier,
                sampler_health_check=self.async_sampler.raise_if_failed,
            )
            self.trainer.add_callback(self.weight_callback)
            logger.info("[7/7] ✓ Callback registered")
            
            # Update state
            self._state = self.SAMPLING
            logger.info("=" * 60)
            logger.info("✓ Orchestrator ready for training (SAMPLING state)")
            logger.info("  - Async sampler running in background")
            logger.info("  - Weights will apply every {} steps".format(self.config.apply_interval))
            logger.info("=" * 60)
            
        except Exception as e:
            self._state = self.FAILED
            if self.async_sampler is not None:
                try:
                    self.async_sampler.stop()
                except Exception as stop_exc:
                    logger.error(f"Error stopping sampler after prepare failure: {stop_exc}")
            logger.error(f"Failed to prepare orchestrator: {e}")
            raise
    
    def run_training(self) -> Dict[str, Any]:
        """
        Execute training with continuous causal weight application.
        
        Ensures background sampler is stopped on exit (even on exceptions)
        using try/finally for resource cleanup.
        
        Returns:
            Training output dict from trainer.train()
            
        Raises:
            ValueError: If prepare() not called first
            Exception: Any exception raised by trainer.train()
        """
        if self._state not in [self.SAMPLING]:
            raise ValueError(
                f"Cannot run training when in {self._state} state. "
                "Call prepare() first."
            )
        
        try:
            self._state = self.TRAINING
            logger.info("Starting causal training with continuous weight application...")

            if self.async_sampler is not None:
                self.async_sampler.raise_if_failed()

            # Optional warm-up before main training
            if self.config.enable_warmup and self.config.warmup_steps:
                logger.info(
                    f"Warm-up enabled. Running {self.config.warmup_steps} warm-up steps before training."
                )
                self.causal_engine.warmup(
                    self._model_ref,
                    self._train_loader_ref,
                    self.config.warmup_steps,
                )
            
            # Execute training (weights applied via callback at each step)
            training_output = self.trainer.train()

            # Marginal likelihood validation after training
            try:
                if hasattr(self.trainer, 'get_eval_dataloader'):
                    eval_loader = self.trainer.get_eval_dataloader()
                    self.causal_engine.validate_marginal_likelihood(
                        self._model_ref,
                        eval_loader,
                        device=self.config.device,
                    )
            except Exception as eval_exc:
                logger.error(
                    f"Marginal likelihood validation failed: {eval_exc}. Continuing."
                )
            
            self._state = self.COMPLETED
            if self.weight_applier is not None:
                logger.info(
                    "Weight application summary: %s",
                    self.weight_applier.get_metrics(),
                )
            logger.info("Causal training completed successfully")
            return training_output
            
        except Exception as e:
            self._state = self.FAILED
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Critical: Always stop background sampler, even on exception
            if self.async_sampler is not None:
                try:
                    self.async_sampler.stop()
                    logger.info("Background sampler stopped")
                except Exception as e:
                    logger.error(f"Error stopping sampler: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive training diagnostics.
        
        Returns dict with causal analysis, budget utilization, and training metrics.
        
        Returns:
            Dict with keys:
            - causal_summary: Causal path analysis
            - budget_utilization: Budget consumption per path (0-1)
            - training_metrics: Final training metrics
            - state: Current orchestrator state
            
        Raises:
            ValueError: If called before training completes
        """
        if self._state not in [self.COMPLETED, self.FAILED]:
            logger.warning(
                f"Getting diagnostics while in {self._state} state "
                "(may contain incomplete data)"
            )
        
        diagnostics = {
            'state': self._state,
            'causal_summary': self.causal_engine.get_causal_summary() 
                if self.causal_engine else None,
            'budget_utilization': self.budget_monitor.get_budget_utilization()
                if self.budget_monitor else None,
            'budget_snapshot': self.budget_monitor.get_current_budget_snapshot()
                if self.budget_monitor else None,
            'budget_monitor_metrics': self.budget_monitor.get_metrics()
                if self.budget_monitor else None,
            'weight_application_metrics': self.weight_applier.get_metrics()
                if self.weight_applier else None,
            'callback_metrics': self.weight_callback.get_metrics()
                if self.weight_callback else None,
            'training_metrics': self.trainer.state.best_metric
                if self.trainer and self.trainer.state else None,
            'async_sampler_status': self.async_sampler.get_status()
                if self.async_sampler else None,
            'callback_error': self.weight_callback.last_error
                if self.weight_callback else None,
            'failure_policy': {
                'causal_gradient_unavailable': 'fail_closed',
                'laplace_phase_failure_notebook': 'fail_closed',
                'generic_orchestrator_exception_notebook': 'fallback_to_standard_lora',
            },
            'config': {
                'total_causal_budget': self.config.total_causal_budget,
                'async_max_steps': self.config.async_max_steps,
                'apply_interval': self.config.apply_interval,
                'device': self.config.device,
                'seed': self.config.seed,
            }
        }
        
        return diagnostics
    
    def reset(self) -> None:
        """
        Reset orchestrator to IDLE state for reuse.
        
        Cleans up resources and allows prepare() to be called again.
        """
        if self.async_sampler is not None:
            try:
                self.async_sampler.stop()
            except Exception as e:
                logger.error(f"Error stopping sampler during reset: {e}")
        
        self.buffer = None
        self.async_sampler = None
        self.weight_applier = None
        self.budget_monitor = None
        self.weight_callback = None
        self._state = self.IDLE
        logger.info("Orchestrator reset to IDLE state")
    
    def __del__(self) -> None:
        """
        Cleanup on object destruction (safety net).
        
        Ensures sampler is stopped if orchestrator is garbage collected.
        """
        if hasattr(self, 'async_sampler') and self.async_sampler is not None:
            try:
                self.async_sampler.stop()
            except Exception:
                pass  # Silent fail on cleanup
