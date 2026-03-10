from unittest import loader

import torch
from peft import TaskType
from src.settings.settings import settings
from src.utils.logger import logger
from src.utils.metrics import UnifiedEvaluator
from src.utils.memory_manager import MemoryOptimizer
from src.finetuner.data_loader import GLUEDataLoader
from src.finetuner.lora_engine import FineTuningEngine as LoRAEngine
from src.finetuner.laplace_engine import LaplaceLoRAEngine
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

def main():
    logger.info("Initializing Fine-tuning Orchestrator...")
    
    # 1. Setup Data and Evaluation 
    loader = GLUEDataLoader(settings.model_id, settings.task_name)
    evaluator = UnifiedEvaluator(settings.task_name)
    val_loader = loader.get_loader("validation", settings.batch_size)
    train_ds = loader.dataset["train"].map(loader._tokenize_fn, batched=True)
    eval_ds = loader.dataset["validation"].map(loader._tokenize_fn, batched=True)
    # 2. Strategy: LoRA / QLoRA Initialization ....
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(settings.model_id)
        # Dependency Injection: Inject configuration into the engine
        lora_engine = LoRAEngine(
            TaskType.SEQ_CLS, base_model, 
            settings.lora_rank, settings.lora_alpha, settings.lora_dropout
        )
        model_lora = lora_engine.apply_lora() # 
        model_lora.to(settings.device) 
        # 3. Training Loop Configuration
        training_args = TrainingArguments(
            output_dir=f"{settings.output_dir}_{settings.task_name}",
            num_train_epochs=settings.epochs,
            per_device_train_batch_size=settings.batch_size,
            remove_unused_columns=False,
            # The 'device' argument is removed as it is not a valid parameter [1]
        )

        # Initialize Trainer with the model already located on the correct device
        trainer = Trainer(
            model=model_lora, 
            args=training_args, 
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=evaluator.compute_all # Use adjusted UnifiedEvaluator
        )

        trainer.train()
        # 4. Bayesian Extension: Laplace-LoRA 
        if settings.execute_laplace:
            # We pass the engine to access adapters for curvature estimation
            laplace_engine = LaplaceLoRAEngine(lora_engine, settings.prior_precision)
            # 8-core CPU friendly: Accumulate curvature iteratively 
            laplace_engine.accumulate_curvature(trainer.get_train_dataloader())            
            # Linearized Prediction for Bayesian Model Averaging 
            predictor = laplace_engine.get_linearized_predictor()
            laplace_logits, labels = predictor.predict_batch(val_loader, settings.device)            
            # Compare with standard metrics 
            laplace_results = evaluator.compute_all(laplace_logits, labels)
            logger.info(f"Laplace-LoRA Results: {laplace_results}")
        # Final cleanup to respect 10GB RAM constraint 
        MemoryOptimizer.cleanup()

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()