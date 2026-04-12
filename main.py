from src.utils.multiprocessing import configure_spawn_context

# Configure multiprocessing as early as possible to avoid fork/spawn mismatches
# in notebook runtimes (e.g., Colab Linux kernels).
_mp_setup = configure_spawn_context()

import os
from pathlib import Path
import torch
from peft import TaskType
from src.settings import SettingsFactory, CausalTrainingConfig
from src.utils.logger import logger
from src.utils.metrics import UnifiedEvaluator
from src.utils.causal_sampler import CausalWeightSampler
from src.finetuner.checkpoint_handler import CheckpointSelector
from src.finetuner.data_loader import GLUEDataLoader
from src.finetuner.lora_engine import FineTuningEngine as LoRAEngine
from src.finetuner.causal_engine import (
    CausalMonteCLoRAEngine,
    EqualBudgetAllocationStrategy,
    TemperatureSoftmaxAllocationStrategy,
)
from src.finetuner.nie_strategy import NIEBudgetAllocationStrategy
from src.finetuner.causal_training_orchestrator import CausalTrainingOrchestrator
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

logger.info(_mp_setup.message)
logger.info(
    "Multiprocessing runtime | requested=%s current=%s env=%s",
    _mp_setup.requested_method,
    _mp_setup.current_method,
    "colab" if _mp_setup.is_colab else "local",
)


def _resume_enabled(policy: str) -> bool:
    """Return True when runtime should attempt strict auto-resume."""
    normalized = (policy or "").strip().lower()
    return normalized in {"latest", "auto", "strict", "true", "1"}


def _resolve_resume_checkpoint(output_dir: str, method: str) -> str | None:
    """Resolve strict auto-resume checkpoint using last-known-good validation only."""
    resume_policy = os.environ.get("RESUME_POLICY", "false")
    if not _resume_enabled(resume_policy):
        logger.info("Resume disabled by policy=%s", resume_policy)
        return None

    selected = CheckpointSelector.select_resume_checkpoint(
        output_dir=Path(output_dir),
        method=method,
    )
    logger.info("Strict resume selected checkpoint: %s", selected.path)
    return str(selected.path)


def main():
    logger.info("Initializing Fine-tuning Orchestrator...")
    settings = SettingsFactory.create_settings(
        override_values={"experiment_type": os.environ.get("EXPERIMENT_TYPE", "lora")}
    )

    # 1. Setup Data and Evaluation
    loader = GLUEDataLoader(
        settings.model_id,
        settings.task_name,
        max_length=settings.max_seq_length,
    )
    evaluator = UnifiedEvaluator(settings.task_name)

    train_ds, eval_ds, _ = loader.get_datasets(train_split="train")

    train_loader = loader.get_loader("train", settings.batch_size)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_t = torch.tensor(logits)
        labels_t = torch.tensor(labels)
        return evaluator.compute_all(logits_t, labels_t)

    # 2. Strategy: LoRA + Causal orchestration
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(settings.model_id)
        lora_engine = LoRAEngine(
            TaskType.SEQ_CLS, base_model,
            settings.lora_rank, settings.lora_alpha, settings.lora_dropout
        )
        model_lora = lora_engine.apply_lora()
        model_lora.to(settings.device)

        # 3. Training Loop Configuration
        training_args = TrainingArguments(
            output_dir=f"{settings.output_dir}_{settings.task_name}",
            num_train_epochs=settings.epochs,
            per_device_train_batch_size=settings.batch_size,
            per_device_eval_batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            remove_unused_columns=False,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model_lora,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        resume_checkpoint = _resolve_resume_checkpoint(
            output_dir=training_args.output_dir,
            method=settings.experiment_type,
        )

        if settings.execute_causal_engine:
            causal_config = CausalTrainingConfig(
                total_causal_budget=1000,
                async_max_steps=100,
                apply_interval=settings.apply_interval,
                device=settings.device,
                enable_warmup=False,
                enable_interventional_weights=True,
                warmup_steps=10,
                causal_softmax_temp_init=settings.causal_softmax_temp_initial,
                causal_softmax_temp_final=settings.causal_softmax_temp_final,
                causal_temp_annealing=(settings.causal_temp_schedule == "linear_decay"),
            )

            # US1: select NIE strategy for causal budgeting in Bayesian mode.
            if settings.causal_sampler_mode == "mixture_of_gaussians":
                budget_strategy = NIEBudgetAllocationStrategy(
                    temp_init=causal_config.causal_softmax_temp_init,
                    temp_final=causal_config.causal_softmax_temp_final,
                    apply_interval=causal_config.apply_interval,
                )
                logger.info(
                    "Using NIEBudgetAllocationStrategy (τ_init=%.4f, τ_final=%.4f, anneal=%s)",
                    causal_config.causal_softmax_temp_init,
                    causal_config.causal_softmax_temp_final,
                    causal_config.causal_temp_annealing,
                )
            else:
                initial_tau = causal_config.get_causal_temperature(progress_ratio=0.0)
                if initial_tau != 1.0 or causal_config.causal_temp_annealing:
                    budget_strategy = TemperatureSoftmaxAllocationStrategy(
                        temperature=initial_tau
                    )
                    logger.info(
                        "Using TemperatureSoftmaxAllocationStrategy (τ_init=%.4f, τ_final=%.4f, anneal=%s)",
                        causal_config.causal_softmax_temp_init,
                        causal_config.causal_softmax_temp_final,
                        causal_config.causal_temp_annealing,
                    )
                else:
                    budget_strategy = EqualBudgetAllocationStrategy()
                    logger.info("Using EqualBudgetAllocationStrategy (τ=1.0 sentinel)")

            causal_engine = CausalMonteCLoRAEngine(
                lora_engine=lora_engine,
                causal_threshold=0.1,
                sample_budget=causal_config.total_causal_budget,
                budget_strategy=budget_strategy,
            )
            if settings.causal_sampler_mode == "mixture_of_gaussians":
                try:
                    from src.utils.bayesian_sampler import BayesianCausalSampler

                    causal_sampler = BayesianCausalSampler(
                        causal_engine=causal_engine,
                        model=model_lora,
                        device=settings.device,
                        enable_pg_pos=settings.enable_pg_pos,
                        kfac_correlation=settings.kfac_correlation,
                        random_dirichlet_init=settings.random_dirichlet_init,
                    )
                    logger.info("Using BayesianCausalSampler (mixture_of_gaussians)")
                except Exception as exc:
                    logger.warning(
                        "Bayesian sampler mode requested but unavailable (%s). Falling back to gradient sampler.",
                        exc,
                    )
                    causal_sampler = CausalWeightSampler(
                        causal_engine=causal_engine,
                        model=model_lora,
                        device=settings.device,
                    )
            else:
                causal_sampler = CausalWeightSampler(
                    causal_engine=causal_engine,
                    model=model_lora,
                    device=settings.device,
                )
                logger.info("Using CausalWeightSampler (gradient)")

            orchestrator = CausalTrainingOrchestrator(
                lora_engine=lora_engine,
                causal_engine=causal_engine,
                trainer=trainer,
                causal_sampler=causal_sampler,
                config=causal_config,
            )

            orchestrator.prepare(model_lora, train_loader)
            orchestrator.run_training(resume_from_checkpoint=resume_checkpoint)
            
            # 4. Save trained LoRA adapter
            output_dir = training_args.output_dir
            trainer.save_model(output_dir)
            lora_engine.save_lora_weights(output_dir)
            logger.info(f"Saved trained LoRA adapter to: {output_dir}")
            
            diagnostics = orchestrator.get_diagnostics()
            logger.info(f"Causal training diagnostics: {diagnostics}")
            logger.info(
                "Causal budget summary | allocation=%s utilization=%s",
                diagnostics.get("budget_snapshot"),
                diagnostics.get("budget_utilization"),
            )
            weight_metrics = diagnostics.get("weight_application_metrics") or {}
            sampler_metrics = (diagnostics.get("async_sampler_status") or {}).get("metrics", {})
            logger.info(
                "Causal runtime summary | state=%s applied=%s skipped_empty=%s sampler_published=%s",
                diagnostics.get("state"),
                weight_metrics.get("times_applied"),
                weight_metrics.get("empty_buffer_skips"),
                sampler_metrics.get("published_batches"),
            )
            return {
                "resume_checkpoint": resume_checkpoint,
                "diagnostics": diagnostics,
                "mode": "causal",
            }
        else:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
            
            # 4. Save trained LoRA adapter for standard LoRA path
            output_dir = training_args.output_dir
            trainer.save_model(output_dir)
            lora_engine.save_lora_weights(output_dir)
            logger.info(f"Saved trained LoRA adapter to: {output_dir}")
            logger.info("Causal engine disabled; completed standard LoRA training.")
            return {
                "resume_checkpoint": resume_checkpoint,
                "diagnostics": None,
                "mode": "standard_lora",
            }

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()