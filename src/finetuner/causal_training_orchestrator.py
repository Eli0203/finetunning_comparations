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

import os
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from src.utils.math_utils import CausalMath
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


class InterventionalWeightCallback(TrainerCallback):
    """Compute bounded interventional weights from a rolling empirical window."""

    def __init__(self, window_size: int = 50, max_weight: float = 10.0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if max_weight <= 0:
            raise ValueError("max_weight must be greater than 0")

        self.window_size = window_size
        self.max_weight = max_weight
        self._feature_window: Deque[int] = deque()
        self._freq_table: Dict[int, int] = {}
        self.applied_steps = 0
        self.last_mean_weight = 0.0

    def _hash_feature(self, input_ids: torch.Tensor) -> int:
        flat_values = tuple(int(value) for value in input_ids.detach().cpu().reshape(-1).tolist())
        return hash(flat_values)

    def _push_feature(self, feature_hash: int) -> None:
        if len(self._feature_window) >= self.window_size:
            evicted = self._feature_window.popleft()
            remaining = self._freq_table[evicted] - 1
            if remaining <= 0:
                del self._freq_table[evicted]
            else:
                self._freq_table[evicted] = remaining

        self._feature_window.append(feature_hash)
        self._freq_table[feature_hash] = self._freq_table.get(feature_hash, 0) + 1

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        del args, state

        inputs = kwargs.get('inputs')
        if not isinstance(inputs, dict):
            return control

        input_ids = inputs.get('input_ids')
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim == 0:
            return control

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        feature_hashes = [self._hash_feature(sample) for sample in input_ids]
        for feature_hash in feature_hashes:
            self._push_feature(feature_hash)

        total = max(len(self._feature_window), 1)
        probabilities = torch.tensor(
            [self._freq_table[feature_hash] / total for feature_hash in feature_hashes],
            dtype=torch.float32,
        )
        p_z = torch.ones(2, dtype=torch.float32) / 2.0
        p_y_given_x_z = probabilities.unsqueeze(-1).repeat(1, 2)
        adjusted = CausalMath.backdoor_adjustment(p_y_given_x_z, p_z).clamp(min=1e-12)
        weights = (1.0 / adjusted).clamp(max=self.max_weight).to(dtype=torch.float32)

        inputs['interventional_weights'] = weights
        self.applied_steps += 1
        self.last_mean_weight = float(weights.mean().item()) if weights.numel() else 0.0
        return control

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'applied_steps': self.applied_steps,
            'window_size': self.window_size,
            'tracked_features': len(self._freq_table),
            'last_mean_weight': self.last_mean_weight,
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
        self.interventional_callback: Optional[InterventionalWeightCallback] = None
        self._model_ref: Optional[nn.Module] = None
        self._train_loader_ref: Optional[Any] = None
        
        # State tracking
        self._state = self.IDLE
        # Warm-up gate tracking (Phase 6)
        self._warmup_complete = False
        self._warmup_gates = {
            'signal': False,
            'loss': False,
            'causal': False,
            'resource': False,
        }
        logger.info("CausalTrainingOrchestrator initialized in IDLE state")
    
    @property
    def state(self) -> str:
        """Return current orchestrator state."""
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        """Allow tests/runtime to set state through a validated setter."""
        valid_states = {
            self.IDLE,
            self.PREPARING,
            self.SAMPLING,
            self.TRAINING,
            self.COMPLETED,
            self.FAILED,
        }
        if value not in valid_states:
            raise ValueError(f"Invalid state: {value}")
        self._state = value

    def _transition_to_sampling(self) -> None:
        """Transition PREPARING -> SAMPLING only after warm-up completion."""
        if self._state not in {self.PREPARING, self.IDLE}:
            raise ValueError(f"Invalid transition to SAMPLING from {self._state}")
        if not self._warmup_complete:
            raise RuntimeError("Cannot transition to SAMPLING before warm-up completion")
        self._state = self.SAMPLING

    def _transition_to_training(self) -> None:
        """Transition SAMPLING -> TRAINING, waiting for in-flight sampler work."""
        if self._state != self.SAMPLING:
            raise ValueError(f"Invalid transition to TRAINING from {self._state}")
        if self.async_sampler is not None and hasattr(self.async_sampler, 'join'):
            self.async_sampler.join()
        self._state = self.TRAINING

    def _transition_to_idle(self) -> None:
        """Transition TRAINING/COMPLETED/FAILED -> IDLE with sampler cleanup."""
        if self.async_sampler is not None and hasattr(self.async_sampler, 'stop'):
            self.async_sampler.stop()
        self._state = self.IDLE

    def _check_warmup_gates(self) -> None:
        """Validate all warm-up gates; raise if any gate is false."""
        failed = [name for name, passed in self._warmup_gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Warm-up gates not satisfied: {failed}")
        self._warmup_complete = True

    def _check_signal_gate(self) -> bool:
        """Signal gate: ||BA||_F > 1e-6."""
        try:
            grads = self.causal_engine.compute_backdoor_gradients()
            if not grads:
                return False
            max_norm = 0.0
            for matrices in grads.values():
                if isinstance(matrices, dict) and 'A' in matrices and 'B' in matrices:
                    norm = torch.linalg.matrix_norm(matrices['A'])
                    max_norm = max(max_norm, float(norm))
            return max_norm > 1e-6
        except Exception:
            return False

    def _check_loss_gate(self) -> bool:
        """Loss gate: recent loss trend should not diverge."""
        try:
            history = getattr(getattr(self.trainer, 'state', None), 'log_history', [])
            losses = [entry.get('loss') for entry in history if isinstance(entry, dict) and 'loss' in entry]
            if len(losses) < 3:
                return True
            recent = losses[-3:]
            return recent[2] <= recent[0]
        except Exception:
            return False

    def _check_causal_gate(self) -> bool:
        """Causal gate: Var(NIE) > 1e-6."""
        try:
            summary = self.causal_engine.get_causal_summary()
            nie_var = float(summary.get('nie_variance', 0.0)) if isinstance(summary, dict) else 0.0
            return nie_var > 1e-6
        except Exception:
            return False

    def _check_resource_gate(self) -> bool:
        """Resource gate: buffer slots available and RAM under configured ceiling."""
        if self.buffer is None or not hasattr(self.buffer, 'available_slots'):
            return False
        if self.buffer.available_slots() <= 0:
            return False

        max_ram = getattr(self.config, 'max_ram_threshold_gb', 9.0)
        if max_ram is None:
            max_ram = 9.0
        try:
            import psutil

            current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            return current_ram < float(max_ram)
        except Exception:
            return True

    def _select_peft_config(self, available_vram_gb: float) -> Dict[str, Any]:
        """Select precision/quantization profile from memory and defaults."""
        settings_obj = getattr(self.config, '_settings', None)
        default_quant = getattr(settings_obj, 'default_quantization', 'auto')

        if default_quant == 'nf4_forced':
            return {'precision': 'nf4', 'quantization': 'nf4'}
        if default_quant == 'fp16':
            return {'precision': 'fp16', 'quantization': 'none'}
        if available_vram_gb < 6.0:
            return {'precision': 'nf4', 'quantization': 'nf4'}
        if available_vram_gb < 12.0:
            return {'precision': 'fp16', 'quantization': 'none'}
        return {'precision': 'fp32', 'quantization': 'none'}

    def _register_peft_config(self, model: Any, peft_config: Dict[str, Any]) -> None:
        """Attach peft configuration to the model object for later consumption."""
        setattr(model, 'peft_config', peft_config)

    def _should_use_nf4(self, available_vram_gb: float, force_nf4: bool = False) -> bool:
        """NF4 selection logic for Low-VRAM and explicit force mode."""
        if force_nf4:
            return True
        settings_obj = getattr(self.config, '_settings', None)
        default_quant = getattr(settings_obj, 'default_quantization', 'auto')
        if default_quant == 'nf4_forced':
            return True
        if default_quant == 'fp16':
            return False
        return available_vram_gb < 6.0

    def _create_bnb_config(self) -> Any:
        """Build a lightweight BitsAndBytes-style config object used by tests/runtime."""
        class _BnbConfig:
            def __init__(self) -> None:
                self.load_in_4bit = True
                self.bnb_4bit_compute_dtype = 'float16'
                self.double_quantization = True
                self.quant_type = 'nf4'

        return _BnbConfig()

    def _load_model_with_bnb(self, model_name: str, bnb_config: Any) -> Any:
        """Load HF model using quantization config."""
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
        )

    def _start_background_sampler(self) -> None:
        """Start async sampler process if available."""
        if self.async_sampler is not None and hasattr(self.async_sampler, 'start'):
            self.async_sampler.start()

    def _stop_background_sampler(self) -> None:
        """Stop async sampler process if available."""
        if self.async_sampler is not None and hasattr(self.async_sampler, 'stop'):
            self.async_sampler.stop()

    def _get_next_weight_async(self) -> Optional[Dict[str, Any]]:
        """Try non-blocking retrieval from async sampler."""
        if self.async_sampler is None or not hasattr(self.async_sampler, 'get_next_weight'):
            return None
        return self.async_sampler.get_next_weight()

    def _get_next_weight_blocking(self) -> Dict[str, Any]:
        """Blocking retrieval when async buffer is empty."""
        if self.async_sampler is None or not hasattr(self.async_sampler, 'get_next_weight'):
            raise RuntimeError("Async sampler is not available")
        candidate = self.async_sampler.get_next_weight()
        if candidate is not None:
            return candidate
        if hasattr(self.async_sampler, 'wait_for_batch'):
            self.async_sampler.wait_for_batch()
        candidate = self.async_sampler.get_next_weight()
        if candidate is None:
            raise RuntimeError("No async weight available after blocking wait")
        return candidate
    
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
            logger.info(f"[1/7] [OK] Identified {len(causal_paths)} causal paths")
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
            logger.info(f"[2/7] [OK] Allocated budget across {len(budget_summary)} paths")
            for path, budget in list(budget_summary.items())[:3]:
                logger.debug(f"      {path}: {budget} samples")

            # Notify the sampler so it can re-read the now-populated budget.
            # This eliminates the "no budget allocation" warning that fires when
            # the sampler is constructed before prepare() is called.
            self.causal_sampler.refresh_path_weights()
            logger.info("[2/7] [OK] Sampler path weights refreshed")
            
            # Step 3: Create double buffer for inter-process communication
            logger.info("[3/7] Creating double buffer for weight communication (O(1) retrieval)...")
            self.buffer = MemoryOptimizer.create_double_buffer()
            logger.info("[3/7] [OK] Double buffer created")
            
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
            logger.info(f"[4/7] [OK] Background sampler started ({self.config.async_max_steps} max steps)")
            
            # Step 5: Create weight applier
            logger.info("[5/7] Creating continuous weight applier (rate-limited application)...")
            self.weight_applier = ContinuousWeightApplier(
                buffer=self.buffer,
                model=model,
                device=self.config.device,
                apply_interval=self.config.apply_interval
            )
            logger.info(f"[5/7] [OK] Weight applier configured (interval: {self.config.apply_interval} steps)")
            
            # Step 6: Create budget monitor
            logger.info("[6/7] Creating training budget monitor (utilization tracking)...")
            self.budget_monitor = TrainingBudgetMonitor(self.causal_engine)
            logger.info("[6/7] [OK] Budget monitor initialized")
            
            # Step 7: Register trainer callback for weight application
            logger.info("[7/7] Registering weight application callback with HuggingFace Trainer...")
            self.weight_callback = WeightApplicationCallback(
                self.weight_applier,
                sampler_health_check=self.async_sampler.raise_if_failed,
            )
            self.trainer.add_callback(self.weight_callback)
            if getattr(self.config, 'enable_interventional_weights', False):
                self.interventional_callback = InterventionalWeightCallback()
                self.trainer.add_callback(self.interventional_callback)
            logger.info("[7/7] [OK] Callback registered")
            
            # Update state
            self._state = self.SAMPLING
            logger.info("=" * 60)
            logger.info("[OK] Orchestrator ready for training (SAMPLING state)")
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
                    mml = self.causal_engine.validate_marginal_likelihood(
                        self._model_ref,
                        eval_loader,
                        device=self.config.device,
                    )
                    if mml is None and self.weight_applier is not None:
                        self.weight_applier.request_skip_next_apply()
                        logger.warning(
                            "Fail-closed marginal likelihood path activated; "
                            "next weight application will be skipped."
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
