import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# Project infrastructure
from src.settings.settings import settings           # Centralized configuration
from src.utils.logger import logger                  # Traceable logging
from src.utils.metrics import compute_glue_metrics   # Standardized GLUE results
from src.utils.memory_manager import MemoryOptimizer # Integrated memory best practices
from src.finetuner.data_loader import GLUEDataLoader # Dataset orchestrator

# Engines (Decoupled via Dependency Injection)
from src.finetuner.lora_engine import FineTuningEngine as LoRAEngine
from src.finetuner.qlora_engine import QLoraEngine

def run_tuning_cycle(mode: str, model, loader: GLUEDataLoader):
    """Orchestrates the fine-tuning and evaluation process for a specific engine."""
    logger.info(f"--- Starting {mode} Cycle for {settings.task_name} ---")
    MemoryOptimizer.log_resource_usage(f"Pre-{mode} Model Initialization")

    # 1. Parameterized TrainingArguments from settings.py
    training_args = TrainingArguments(
        output_dir=f"{settings.output_dir}_{mode}",
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        eval_strategy=settings.eval_strategy,         # Renamed from evaluation_strategy
        save_strategy=settings.save_strategy,
        save_total_limit=settings.save_total_limit,
        logging_dir=settings.logging_dir,
        logging_strategy=settings.logging_strategy,
        fp16=torch.cuda.is_available(),               # Auto-detect GPU acceleration
        remove_unused_columns=False                   # Required for BERT input passing
    )

    # 2. Dataset Formatting for BERT [modeling_bert.py:667 fix]
    # Mapping the tokenize function and forcing torch tensor output
    train_ds = loader.dataset["train"].map(loader._tokenize_fn, batched=True)
    eval_ds = loader.dataset["validation"].map(loader._tokenize_fn, batched=True)
    
    cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    train_ds.set_format(type="torch", columns=cols)
    eval_ds.set_format(type="torch", columns=cols)

    # 3. Initialize Trainer with injected custom metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda p: compute_glue_metrics(
            settings.task_name, torch.tensor(p.predictions), torch.tensor(p.label_ids)
        )
    )

    # 4. Execution & Persistence
    logger.info(f"Executing {mode} on {settings.device}...")
    trainer.train()
    results = trainer.evaluate()
    trainer.save_model(f"./final_model_{mode}")
    
    logger.info(f"{mode} results summary: {results}")
    MemoryOptimizer.log_resource_usage(f"Post-{mode} Completion")
    return results

def main():
    """Main application entry point following the DI pattern [2]."""
    logger.info(f"System initialized for device: {settings.device}")
    
    if settings.use_mock_data:
        logger.warning("Running in MOCK mode. Parameters loaded but training bypassed.")
        return

    # Initialize Data Orchestrator
    loader = GLUEDataLoader(settings.model_id, settings.task_name)

    # 4.1.1 Logic for LoRA Method
    if settings.execute_lora:
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(settings.model_id)
            # WIRING: Inject parameters into engine constructor
            engine = LoRAEngine(
                task_type=settings.task_type,
                model=base_model,
                rank=settings.lora_rank,
                alpha=settings.lora_alpha,
                dropout=settings.lora_dropout
            )
            model = engine.apply_lora()
            run_tuning_cycle("LoRA", model, loader)
            
            # Aggressive cleanup for 10GB limit
            del model, base_model
            MemoryOptimizer.cleanup()
        except Exception as e:
            logger.error(f"LoRA execution failed: {e}", exc_info=True)

    # 4.1.2 Logic for QLoRA Method
    if settings.execute_qlora:
        try:
            # QLoRA handles its own quantized loading internally
            engine = QLoraEngine(
                model_id=settings.model_id,
                rank=settings.qlora_rank,
                alpha=settings.qlora_alpha,
                dropout=settings.qlora_dropout
            )
            model = engine.prepare_model()
            run_tuning_cycle("QLoRA", model, loader)
            
            del model
            MemoryOptimizer.cleanup()
        except Exception as e:
            logger.error(f"QLoRA execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()