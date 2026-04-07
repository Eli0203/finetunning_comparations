from src.utils.multiprocessing import configure_spawn_context

# Configure multiprocessing as early as possible to avoid fork/spawn mismatches
# in notebook runtimes (e.g., Colab Linux kernels).
_mp_setup = configure_spawn_context()

import torch
from peft import TaskType
from src.settings.settings import settings
from src.utils.logger import logger
from src.utils.metrics import UnifiedEvaluator
from src.utils.causal_sampler import CausalWeightSampler
from src.finetuner.data_loader import GLUEDataLoader
from src.finetuner.lora_engine import FineTuningEngine as LoRAEngine
from src.finetuner.causal_engine import (
    CausalMonteCLoRAEngine,
    EqualBudgetAllocationStrategy,
    TemperatureSoftmaxAllocationStrategy,
)
from src.finetuner.causal_training_orchestrator import CausalTrainingOrchestrator
from src.settings.settings import CausalTrainingConfig
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

logger.info(_mp_setup.message)
logger.info(
    "Multiprocessing runtime | requested=%s current=%s env=%s",
    _mp_setup.requested_method,
    _mp_setup.current_method,
    "colab" if _mp_setup.is_colab else "local",
)


def main():
    logger.info("Initializing Fine-tuning Orchestrator...")

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
        )

        trainer = Trainer(
            model=model_lora,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        if settings.execute_causal_engine:
            causal_config = CausalTrainingConfig(
                total_causal_budget=1000,
                async_max_steps=100,
                apply_interval=10,
                device=settings.device,
                enable_warmup=False,
                warmup_steps=10,
            )

            # Select budget allocation strategy: temperature != 1.0 activates
            # temperature-scaled softmax; 1.0 (default) uses equal allocation.
            if causal_config.temperature != 1.0:
                budget_strategy = TemperatureSoftmaxAllocationStrategy(
                    temperature=causal_config.temperature
                )
                logger.info(
                    "Using TemperatureSoftmaxAllocationStrategy (τ=%.4f, anneal=%s)",
                    causal_config.temperature,
                    causal_config.temperature_anneal,
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
            causal_sampler = CausalWeightSampler(
                causal_engine=causal_engine,
                model=model_lora,
                device=settings.device,
            )

            orchestrator = CausalTrainingOrchestrator(
                lora_engine=lora_engine,
                causal_engine=causal_engine,
                trainer=trainer,
                causal_sampler=causal_sampler,
                config=causal_config,
            )

            orchestrator.prepare(model_lora, train_loader)
            orchestrator.run_training()
            diagnostics = orchestrator.get_diagnostics()
            logger.info(f"Causal training diagnostics: {diagnostics}")
            weight_metrics = diagnostics.get("weight_application_metrics") or {}
            sampler_metrics = (diagnostics.get("async_sampler_status") or {}).get("metrics", {})
            logger.info(
                "Causal runtime summary | state=%s applied=%s skipped_empty=%s sampler_published=%s",
                diagnostics.get("state"),
                weight_metrics.get("times_applied"),
                weight_metrics.get("empty_buffer_skips"),
                sampler_metrics.get("published_batches"),
            )
        else:
            trainer.train()
            logger.info("Causal engine disabled; completed standard LoRA training.")

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()