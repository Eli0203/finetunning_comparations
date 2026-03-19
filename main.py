import torch
from peft import TaskType
from src.settings.settings import settings
from src.utils.logger import logger
from src.utils.metrics import UnifiedEvaluator
from src.utils.causal_sampler import CausalWeightSampler
from src.finetuner.data_loader import GLUEDataLoader
from src.finetuner.lora_engine import FineTuningEngine as LoRAEngine
from src.finetuner.causal_engine import CausalMonteCLoRAEngine
from src.finetuner.causal_training_orchestrator import CausalTrainingOrchestrator
from src.settings.settings import CausalTrainingConfig
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


def main():
    logger.info("Initializing Fine-tuning Orchestrator...")

    # 1. Setup Data and Evaluation
    loader = GLUEDataLoader(settings.model_id, settings.task_name)
    evaluator = UnifiedEvaluator(settings.task_name)

    train_ds = loader.dataset["train"].map(loader._tokenize_fn, batched=True)
    eval_ds = loader.dataset["validation"].map(loader._tokenize_fn, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

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
            causal_engine = CausalMonteCLoRAEngine(
                lora_engine=lora_engine,
                causal_threshold=0.1,
                sample_budget=1000,
            )
            causal_sampler = CausalWeightSampler(
                causal_engine=causal_engine,
                model=model_lora,
                device=settings.device,
            )
            causal_config = CausalTrainingConfig(
                total_causal_budget=1000,
                async_max_steps=100,
                apply_interval=10,
                device=settings.device,
                enable_warmup=False,
                warmup_steps=10,
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
        else:
            trainer.train()
            logger.info("Causal engine disabled; completed standard LoRA training.")

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()