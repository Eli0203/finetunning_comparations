"""
Causal Monte Carlo LoRA Engine
Integrates causal inference with LoRA fine-tuning for Bayesian optimization.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from src.finetuner.lora_engine import FineTuningEngine
from src.utils.math_utils import LaplaceMath
from src.utils.logger import logger


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
        sample_budget: int = 1000
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
        self.causal_paths: List[str] = []
        self.budget_allocation: Dict[str, int] = {}
        self._warmup_state: Dict[str, Any] = {
            'enabled': False,
            'completed': False,
            'steps': 0,
            'loss_trajectory': [],
        }
        self._marginal_likelihood: Optional[float] = None

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
        logger.info("Identifying causal paths using backdoor adjustment...")

        device = self._resolve_device(model)
        logger.debug(f"Sensitivity analysis device: {device}")

        # --- Populate gradients via a single forward/backward pass ---
        # train() mode is required so that BN/Dropout behave correctly and
        # gradients flow through all parameters.
        model.train()
        model.zero_grad()
        gradients_populated = False

        for batch in data_loader:
            try:
                filtered = self._filter_model_inputs(batch)
                if not filtered:
                    logger.debug("Batch produced no valid model inputs; skipping.")
                    break
                filtered = self._move_batch_to_device(filtered, device)
                outputs = model(**filtered)
                loss = getattr(outputs, 'loss', None)
                if loss is not None:
                    loss.backward()
                    gradients_populated = True
                else:
                    logger.debug("Batch has no loss output; gradients not computed.")
            except Exception as exc:
                logger.warning(f"Sensitivity forward pass failed: {exc}. Returning no causal paths.")
            break  # A single batch is sufficient for gradient-based sensitivity

        if not gradients_populated:
            logger.warning(
                "Gradient population skipped (empty loader, no loss, or forward error). "
                "Returning empty causal paths."
            )

        model.eval()

        # --- Inspect per-module gradient magnitudes ---
        causal_paths = []
        for name, module in model.named_modules():
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            sensitivity = self._compute_causal_sensitivity(module)
            if sensitivity > self.causal_threshold:
                causal_paths.append(name)
                logger.debug(f"Module {name} shows causal sensitivity: {sensitivity:.4f}")

        # Clean up gradients so training loop starts from scratch
        model.zero_grad()

        self.causal_paths = causal_paths
        logger.info(f"Identified {len(causal_paths)} causal paths: {causal_paths}")
        return causal_paths

    def allocate_budget(self, causal_paths: List[str], total_budget: int) -> Dict[str, int]:
        """
        Allocate sample budget across causal paths using mediation analysis.

        Args:
            causal_paths: List of causal path names
            total_budget: Total samples to allocate

        Returns:
            Dictionary mapping path names to allocated sample counts
        """
        logger.info(f"Allocating budget of {total_budget} samples across {len(causal_paths)} causal paths")

        if not causal_paths:
            logger.warning("No causal paths provided, using equal allocation")
            return {}

        # For now, use equal allocation - can be enhanced with NIE scores later
        base_allocation = total_budget // len(causal_paths)
        remainder = total_budget % len(causal_paths)

        allocation = {}
        for i, path in enumerate(causal_paths):
            # Distribute remainder across first few paths
            extra = 1 if i < remainder else 0
            allocation[path] = base_allocation + extra

        self.budget_allocation = allocation
        logger.info(f"Budget allocation: {allocation}")

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
        }

    def warmup(
        self,
        model: nn.Module,
        train_loader,
        num_warmup_steps: int,
        lr: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Run a lightweight MAP warm-up phase before main training.

        Args:
            model: Model to warm up.
            train_loader: Training loader yielding dict-like batches.
            num_warmup_steps: Number of warm-up updates to perform.
            lr: Optimizer learning rate for warm-up.

        Returns:
            Warm-up diagnostics dictionary.
        """
        if num_warmup_steps <= 0:
            self._warmup_state = {
                'enabled': False,
                'completed': False,
                'steps': 0,
                'loss_trajectory': [],
            }
            return self._warmup_state

        model.train()
        device = self._resolve_device(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses: List[float] = []
        steps = 0

        logger.info(f"Starting warm-up for {num_warmup_steps} steps (device={device})")
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
                except Exception as exc:
                    logger.error(f"Warm-up step failed: {exc}. Continuing.")

            # Stop if loader iteration made no progress.
            if steps == steps_before_round:
                break

        self._warmup_state = {
            'enabled': True,
            'completed': steps > 0,
            'steps': steps,
            'loss_trajectory': losses,
        }
        logger.info(f"Warm-up finished with {steps} completed steps")
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