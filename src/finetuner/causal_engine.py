"""
Causal Monte Carlo LoRA Engine
Integrates causal inference with LoRA fine-tuning for Bayesian optimization.
"""

import inspect
import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Protocol
from src.finetuner.lora_engine import FineTuningEngine
from src.utils.math_utils import LaplaceMath
from src.utils.logger import logger


class CausalGradientUnavailableError(RuntimeError):
    """Raised when causal path detection cannot populate gradients."""


class BudgetAllocationStrategy(Protocol):
    """Strategy contract for allocating a sampling budget across causal paths."""

    def allocate(self, causal_paths: List[str], total_budget: int) -> Dict[str, int]:
        """Return budget assignments for each causal path."""


class EqualBudgetAllocationStrategy:
    """Default budget allocation strategy with even distribution and remainder balancing."""

    def allocate(self, causal_paths: List[str], total_budget: int) -> Dict[str, int]:
        if total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        if not causal_paths:
            return {}

        base_allocation = total_budget // len(causal_paths)
        remainder = total_budget % len(causal_paths)
        return {
            path: base_allocation + (1 if idx < remainder else 0)
            for idx, path in enumerate(causal_paths)
        }


def _log_sum_exp(values: List[float]) -> float:
    """Numerically stable log-sum-exp using stdlib math only.

    Args:
        values: List of float values (e.g., score / τ).

    Returns:
        log(Σ exp(v)) computed without overflow.
    """
    if not values:
        return 0.0
    max_val = max(values)
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


class AnnealingTemperatureScheduler:
    """
    Single responsibility: compute the softmax temperature τ at a given training step.

    Linearly decays τ from *initial_temperature* to *temperature_min* over
    *total_steps* training steps.  When annealing is disabled, always returns
    *initial_temperature* regardless of the current step.

    Does NOT hold any reference to a model, engine, or data loader.
    """

    def __init__(
        self,
        initial_temperature: float,
        temperature_min: float = 0.1,
        total_steps: int = 1,
        enabled: bool = False,
    ) -> None:
        """
        Initialise the scheduler.

        Args:
            initial_temperature: Starting τ value (must be > 0).
            temperature_min: Floor τ value for the annealing schedule (must be > 0
                and < initial_temperature when enabled=True).
            total_steps: Total training steps over which to anneal (must be >= 1).
            enabled: When False the scheduler is a no-op and always returns
                *initial_temperature*.

        Raises:
            ValueError: On invalid parameter combinations.
        """
        if initial_temperature <= 0:
            raise ValueError(
                f"initial_temperature must be > 0, got {initial_temperature}"
            )
        if temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be > 0, got {temperature_min}"
            )
        if enabled and temperature_min >= initial_temperature:
            raise ValueError(
                f"temperature_min ({temperature_min}) must be < initial_temperature "
                f"({initial_temperature}) when enabled=True."
            )
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")

        self._initial = initial_temperature
        self._min = temperature_min
        self._total = total_steps
        self._enabled = enabled
        logger.debug(
            "AnnealingTemperatureScheduler initialised "
            "(τ_init=%.4f, τ_min=%.4f, total_steps=%d, enabled=%s)",
            initial_temperature, temperature_min, total_steps, enabled,
        )

    def get_temperature(self, step: int) -> float:
        """
        Compute τ at the given training step.

        Args:
            step: Current training step (0-indexed).  Steps beyond *total_steps*
                are clamped to *temperature_min*.

        Returns:
            Temperature τ ∈ [temperature_min, initial_temperature].
        """
        if not self._enabled:
            return self._initial
        fraction = min(1.0, max(0.0, step / self._total))
        tau = self._initial - (self._initial - self._min) * fraction
        return max(self._min, tau)

    @property
    def initial_temperature(self) -> float:
        """Starting temperature value."""
        return self._initial

    @property
    def temperature_min(self) -> float:
        """Floor temperature value."""
        return self._min

    @property
    def enabled(self) -> bool:
        """Whether annealing is active."""
        return self._enabled


class TemperatureSoftmaxAllocationStrategy:
    """
    Budget allocation via temperature-scaled softmax over raw gradient scores.

    Formula: weight_i = exp(score_i / τ) / Σ_j exp(score_j / τ)
             budget_i = largest-remainder rounding of weight_i * total_budget

    Uses numerically stable log-sum-exp via stdlib ``math`` — no new
    dependencies.  τ = 1.0 gives uniform softmax when all scores are equal;
    for exact equal splits regardless of scores use ``EqualBudgetAllocationStrategy``.

    Satisfies the ``BudgetAllocationStrategy`` Protocol: the extra ``scores``
    keyword argument is optional and the two-argument call signature is valid.

    Does NOT hold any reference to a model or the causal engine.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Initialise the strategy.

        Args:
            temperature: Softmax temperature τ (must be > 0).  Lower values
                concentrate budget on high-scoring paths; higher values spread
                it more uniformly.

        Raises:
            ValueError: If temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._temperature = temperature
        logger.debug(
            "TemperatureSoftmaxAllocationStrategy initialised (τ=%.4f)", temperature
        )

    @property
    def temperature(self) -> float:
        """Current temperature τ."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"temperature must be > 0, got {value}")
        self._temperature = value

    def allocate(
        self,
        causal_paths: List[str],
        total_budget: int,
        scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        """
        Allocate *total_budget* samples proportional to temperature-scaled softmax weights.

        Args:
            causal_paths: Ordered list of causal path names.
            total_budget: Total samples to distribute (must be >= 0).
            scores: Optional mapping of path name → raw score (e.g. gradient
                magnitude).  Paths absent from *scores* receive a score of 0.0.
                When *scores* is None or empty, all paths receive score 0.0 which
                yields a uniform softmax allocation.

        Returns:
            Dict mapping each path to a non-negative integer sample count.
            The values are guaranteed to sum to *total_budget*.

        Raises:
            ValueError: If total_budget < 0.
        """
        if total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        if not causal_paths:
            return {}
        if total_budget == 0:
            return {path: 0 for path in causal_paths}

        # Build score vector (missing paths default to 0.0)
        raw_scores = [
            (scores.get(path, 0.0) if scores else 0.0)
            for path in causal_paths
        ]

        # Temperature-scaled softmax with numerically stable log-sum-exp
        tau = self._temperature
        scaled = [s / tau for s in raw_scores]
        log_sum = _log_sum_exp(scaled)
        softmax_weights = [math.exp(s - log_sum) for s in scaled]

        # Largest-remainder method guarantees sum(budgets) == total_budget
        raw_budgets = [w * total_budget for w in softmax_weights]
        floor_budgets = [int(b) for b in raw_budgets]
        remainders = [r - f for r, f in zip(raw_budgets, floor_budgets)]

        deficit = total_budget - sum(floor_budgets)
        indices_by_remainder = sorted(
            range(len(remainders)), key=lambda i: remainders[i], reverse=True
        )
        for i in range(deficit):
            floor_budgets[indices_by_remainder[i]] += 1

        allocation = {path: floor_budgets[idx] for idx, path in enumerate(causal_paths)}
        logger.debug(
            "TemperatureSoftmaxAllocationStrategy allocated (τ=%.4f): %s",
            tau, allocation,
        )
        return allocation


class CausalMonteCLoRAEngine:
    """
    Causal orchestration engine that integrates causal inference with LoRA fine-tuning.

    Uses dependency injection to accept a LoRA engine and applies causal analysis
    to optimize fine-tuning decisions.
    """

    def __init__(
        self,
        lora_engine: FineTuningEngine,
        causal_threshold: float = 0.1,
        sample_budget: int = 1000,
        budget_strategy: Optional[BudgetAllocationStrategy] = None,
    ):
        """
        Initialize the causal engine with dependency injection.

        Args:
            lora_engine: Injected LoRA engine instance
            causal_threshold: Minimum causal sensitivity to consider a path
            sample_budget: Total samples to allocate across causal paths
        """
        self.lora_engine = lora_engine
        self.causal_threshold = causal_threshold
        self.sample_budget = sample_budget
        self._budget_strategy = budget_strategy or EqualBudgetAllocationStrategy()
        self.causal_paths: List[str] = []
        self.budget_allocation: Dict[str, int] = {}
        self._warmup_state: Dict[str, Any] = {
            'enabled': False,
            'completed': False,
            'steps': 0,
            'loss_trajectory': [],
        }
        self._marginal_likelihood: Optional[float] = None
        self._last_identification_error: Optional[str] = None

        logger.info(f"CausalMonteCLoRAEngine initialized with threshold={causal_threshold}, budget={sample_budget}")

    def identify_causal_paths(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> List[str]:
        """
        Identify modules with significant causal sensitivity using backdoor adjustment.

        Sensitivity is approximated by gradient magnitude: one forward + backward pass
        is executed on a single mini-batch (device-aware) to populate ``.grad`` on each
        module's weight, then modules whose gradient L1 mean exceeds
        ``causal_threshold`` are selected as causal paths.

        Args:
            model: The model to analyse.  Placed on its current device automatically.
            data_loader: Data loader used for gradient computation.

        Returns:
            List of module names that show causal sensitivity above threshold.
        """
        logger.info("=" * 70)
        logger.info("PHASE: Identifying causal paths using backdoor adjustment...")
        logger.info("=" * 70)

        device = self._resolve_device(model)
        self._last_identification_error = None
        logger.debug(f"Sensitivity analysis device: {device}")
        logger.debug(f"Causal threshold: {self.causal_threshold}")

        # --- Populate gradients via a single forward/backward pass ---
        # train() mode is required so that BN/Dropout behave correctly and
        # gradients flow through all parameters.
        model.train()
        model.zero_grad()

        try:
            batch = next(iter(data_loader))
        except StopIteration as exc:
            message = (
                "Gradient population failed: training loader is empty or exhausted. "
                "Baseline LoRA fallback is disabled for causal execution."
            )
            self._last_identification_error = message
            logger.error(message)
            raise CausalGradientUnavailableError(message) from exc
        except Exception as exc:
            message = (
                f"Gradient population failed while reading from the training loader: {exc}. "
                "Baseline LoRA fallback is disabled for causal execution."
            )
            self._last_identification_error = message
            logger.error(message)
            raise CausalGradientUnavailableError(message) from exc

        if isinstance(batch, dict):
            logger.debug(f"Batch keys: {list(batch.keys())}")

        try:
            filtered = self._filter_model_inputs(batch)
            if not filtered:
                raise CausalGradientUnavailableError(
                    "Gradient population failed: batch produced no valid model inputs. "
                    "Baseline LoRA fallback is disabled for causal execution."
                )

            filtered = self._move_batch_to_device(filtered, device)
            outputs = model(**filtered)
            loss = getattr(outputs, 'loss', None)
            if loss is None:
                raise CausalGradientUnavailableError(
                    "Gradient population failed: model outputs do not expose a loss tensor. "
                    "Baseline LoRA fallback is disabled for causal execution."
                )

            loss.backward()
            logger.debug(f"Gradient population succeeded with loss={float(loss.detach().item()):.6f}")
        except CausalGradientUnavailableError as exc:
            self._last_identification_error = str(exc)
            model.zero_grad()
            logger.error(self._last_identification_error)
            raise
        except Exception as exc:
            message = (
                f"Gradient population failed during the causal sensitivity pass: {exc}. "
                "Baseline LoRA fallback is disabled for causal execution."
            )
            self._last_identification_error = message
            model.zero_grad()
            logger.error(message)
            raise CausalGradientUnavailableError(message) from exc

        model.eval()

        # --- Inspect per-module gradient magnitudes ---
        causal_paths = []
        module_sensitivities = []
        for name, module in model.named_modules():
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            sensitivity = self._compute_causal_sensitivity(module)
            module_sensitivities.append((name, sensitivity))
            if sensitivity > self.causal_threshold:
                causal_paths.append(name)
                logger.debug(f"Module {name} shows causal sensitivity: {sensitivity:.4f}")

        if module_sensitivities:
            logger.debug("Top 10 most sensitive modules:")
            for name, sensitivity in sorted(module_sensitivities, key=lambda item: item[1], reverse=True)[:10]:
                logger.debug(f"  {name}: {sensitivity:.4f}")

        # Clean up gradients so training loop starts from scratch
        model.zero_grad()

        self.causal_paths = causal_paths
        if causal_paths:
            logger.info(f"Identified {len(causal_paths)} causal paths: {causal_paths}")
        else:
            logger.warning(
                "No causal paths identified above threshold after successful gradient population. "
                "Causal execution can continue only if the caller explicitly accepts empty allocation."
            )
        return causal_paths

    def allocate_budget(
        self,
        causal_paths: List[str],
        total_budget: int,
        scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        """
        Allocate sample budget across causal paths using mediation analysis.

        Dispatches to the configured ``BudgetAllocationStrategy``.  When *scores*
        are provided and the strategy supports them (i.e. its ``allocate`` method
        accepts a ``scores`` keyword argument), the scores are forwarded.
        Otherwise a warning is logged and the strategy is called without scores.

        Args:
            causal_paths: List of causal path names.
            total_budget: Total samples to allocate.
            scores: Optional dict mapping path name → raw gradient magnitude or
                other causal-importance score.  Forwarded to score-aware
                strategies only.

        Returns:
            Dictionary mapping path names to allocated sample counts.
        """
        logger.info(
            "Allocating budget of %d samples across %d causal paths",
            total_budget, len(causal_paths),
        )

        if not causal_paths:
            logger.warning("No causal paths provided, using equal allocation")
            return {}

        # Detect whether the configured strategy accepts a 'scores' parameter.
        sig = inspect.signature(self._budget_strategy.allocate)
        strategy_accepts_scores = 'scores' in sig.parameters

        if scores is not None and not strategy_accepts_scores:
            logger.warning(
                "scores provided but %s.allocate() does not accept a 'scores' "
                "parameter; falling back to score-free allocation.",
                type(self._budget_strategy).__name__,
            )

        if strategy_accepts_scores and scores is not None:
            allocation = self._budget_strategy.allocate(
                causal_paths, total_budget, scores=scores
            )
        else:
            allocation = self._budget_strategy.allocate(causal_paths, total_budget)

        self.budget_allocation = allocation
        logger.info("Budget allocation: %s", allocation)
        return allocation

    def _compute_causal_sensitivity(self, module: nn.Module) -> float:
        """
        Read the L1 gradient magnitude for *module.weight* after a backward pass.

        ``identify_causal_paths`` is responsible for running the forward/backward
        pass that populates ``.grad`` before this method is called.  If no
        gradient is available (e.g. the loader was empty) this returns ``0.0``
        so the module is safely excluded from the causal paths.

        Args:
            module: The sub-module to inspect.

        Returns:
            Mean absolute gradient value, or 0.0 if no gradient is present.
        """
        if not hasattr(module, 'weight') or module.weight is None:
            return 0.0
        if module.weight.grad is None:
            return 0.0
        return float(module.weight.grad.abs().mean().item())

    def get_causal_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal analysis results.

        Returns:
            Dictionary with causal paths and budget allocation
        """
        return {
            'causal_paths': self.causal_paths,
            'budget_allocation': self.budget_allocation,
            'total_budget': self.sample_budget,
            'causal_threshold': self.causal_threshold,
            'warmup_state': self.get_warmup_state(),
            'marginal_likelihood': self._marginal_likelihood,
            'last_identification_error': self._last_identification_error,
        }

    def warmup(
        self,
        model: nn.Module,
        train_loader,
        num_warmup_steps: int,
        lr: float = 1e-5,
        plateau_delta: float = 1e-4,
        plateau_patience: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a lightweight MAP warm-up phase before main training.

        Supports early exit when the training loss plateaus for
        *plateau_patience* consecutive steps without improving by at least
        *plateau_delta*.  The plateau counter resets whenever a sufficient
        improvement is observed, so transient fluctuations do not trigger
        premature exit.

        Args:
            model: Model to warm up.
            train_loader: Training loader yielding dict-like batches.
            num_warmup_steps: Maximum number of warm-up updates to perform.
            lr: Optimizer learning rate for warm-up.
            plateau_delta: Minimum absolute loss decrease required to reset the
                plateau counter.  Set to 0 to disable early exit entirely.
            plateau_patience: Number of consecutive non-improving steps that
                trigger early exit.  Must be >= 1.

        Returns:
            Warm-up diagnostics dictionary with keys:
                - ``enabled`` (bool): True when steps > 0 were requested.
                - ``completed`` (bool): True when at least one step ran.
                - ``steps`` (int): Actual number of steps executed.
                - ``loss_trajectory`` (List[float]): Per-step loss values.
                - ``early_exit_triggered`` (bool): True if plateau caused early stop.
                - ``plateau_detected_at_step`` (Optional[int]): Step index (1-based)
                  at which the plateau criterion was met, or None.
        """
        if num_warmup_steps <= 0:
            self._warmup_state = {
                'enabled': False,
                'completed': False,
                'steps': 0,
                'loss_trajectory': [],
                'early_exit_triggered': False,
                'plateau_detected_at_step': None,
            }
            return self._warmup_state

        model.train()
        device = self._resolve_device(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses: List[float] = []
        steps = 0

        # Plateau-detection state
        best_loss: Optional[float] = None
        patience_counter: int = 0
        early_exit_triggered: bool = False
        plateau_detected_at_step: Optional[int] = None

        logger.info(
            "Starting warm-up for %d steps (device=%s, plateau_delta=%.2e, plateau_patience=%d)",
            num_warmup_steps, device, plateau_delta, plateau_patience,
        )

        while steps < num_warmup_steps:
            steps_before_round = steps
            for batch in train_loader:
                if steps >= num_warmup_steps:
                    break

                try:
                    optimizer.zero_grad(set_to_none=True)
                    filtered = self._filter_model_inputs(batch)
                    filtered = self._move_batch_to_device(filtered, device)
                    outputs = model(**filtered)
                    loss = getattr(outputs, 'loss', None)

                    if loss is None:
                        logger.debug("Warm-up batch has no loss. Skipping step.")
                        continue

                    loss.backward()
                    optimizer.step()

                    loss_value = float(loss.detach().item())
                    losses.append(loss_value)
                    steps += 1

                    # --- Plateau detection ---
                    if best_loss is None:
                        best_loss = loss_value
                    elif best_loss - loss_value >= plateau_delta:
                        # Sufficient improvement: reset counter and update best
                        best_loss = loss_value
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= plateau_patience:
                            early_exit_triggered = True
                            plateau_detected_at_step = steps
                            logger.info(
                                "Warm-up early exit at step %d: "
                                "loss plateau detected (patience=%d, delta=%.2e).",
                                steps, plateau_patience, plateau_delta,
                            )
                            break

                except Exception as exc:
                    logger.error(f"Warm-up step failed: {exc}. Continuing.")

            # Exit outer loop if plateau triggered or no progress was made
            if early_exit_triggered:
                break
            if steps == steps_before_round:
                break

        self._warmup_state = {
            'enabled': True,
            'completed': steps > 0,
            'steps': steps,
            'loss_trajectory': losses,
            'early_exit_triggered': early_exit_triggered,
            'plateau_detected_at_step': plateau_detected_at_step,
        }
        logger.info(
            "Warm-up finished: steps=%d, early_exit=%s",
            steps, early_exit_triggered,
        )
        return self._warmup_state

    def get_warmup_state(self) -> Dict[str, Any]:
        """Return warm-up diagnostics."""
        return self._warmup_state

    def validate_marginal_likelihood(self, model: nn.Module, val_loader, device: str = 'cpu') -> float:
        """
        Estimate marginal likelihood using Laplace approximation utilities.

        Args:
            model: Model to evaluate.
            val_loader: Validation loader.
            device: Evaluation device.

        Returns:
            Estimated marginal likelihood value.
        """
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    filtered_batch = self._filter_model_inputs(batch)
                    # Always move to target device regardless of whether it is CPU or CUDA
                    filtered_batch = self._move_batch_to_device(filtered_batch, device)

                    outputs = model(**filtered_batch)
                    loss = getattr(outputs, 'loss', None)
                    if loss is None:
                        continue

                    total_loss += float(loss.item())
                    count += 1
                except Exception as exc:
                    logger.error(f"Validation batch failed during MML computation: {exc}")

        avg_neg_log_likelihood = (total_loss / max(1, count))
        log_likelihood = -avg_neg_log_likelihood

        precision_diag = []
        for param in model.parameters():
            if param.requires_grad:
                precision_diag.append(param.detach().abs().flatten() + 1e-6)

        if precision_diag:
            precision_diag_tensor = torch.cat(precision_diag)
        else:
            precision_diag_tensor = torch.ones(1)

        n_params = int(precision_diag_tensor.numel())
        mml = float(LaplaceMath.model_evidence(log_likelihood, precision_diag_tensor, n_params))

        if torch.isnan(torch.tensor(mml)) or torch.isinf(torch.tensor(mml)):
            logger.warning("Marginal likelihood is NaN/Inf; check data and warm-up stability.")

        self._marginal_likelihood = mml
        logger.info(f"Estimated marginal likelihood: {mml:.6f}")
        return mml

    @staticmethod
    def _resolve_device(model: nn.Module) -> torch.device:
        """Return the device of the first model parameter, defaulting to CPU.

        Args:
            model: Any ``nn.Module``.

        Returns:
            The ``torch.device`` where the model lives.
        """
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    @staticmethod
    def _move_batch_to_device(
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Move all tensor values in *batch* to *device*.

        Non-tensor values (e.g. strings, ints) are left unchanged.

        Args:
            batch: A dict of keyword arguments for a model's forward method.
            device: Target ``torch.device``.

        Returns:
            A new dict with tensor values moved to *device*.
        """
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _filter_model_inputs(batch: Any) -> Dict[str, Any]:
        """Filter common dataloader keys to valid model forward inputs."""
        if not isinstance(batch, dict):
            return {}

        allowed = {'input_ids', 'attention_mask', 'token_type_ids', 'labels'}
        filtered = {}
        for key, value in batch.items():
            if key == 'label':
                filtered['labels'] = value
            elif key in allowed:
                filtered[key] = value
        return filtered