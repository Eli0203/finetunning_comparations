import torch
from src.finetuner.causal_engine import CausalMonteCLoRAEngine
from src.finetuner.lora_engine import FineTuningEngine
from peft import TaskType


def test_causal_engine_injection():
    """Test that CausalMonteCLoRAEngine accepts LoRA engine via dependency injection."""
    # Create mock base model
    mock_model = torch.nn.Linear(10, 2)

    # Create LoRA engine
    lora_engine = FineTuningEngine(
        task_type=TaskType.SEQ_CLS,
        model=mock_model,
        rank=4,
        alpha=8,
        dropout=0.1
    )

    # Inject into causal engine
    causal_engine = CausalMonteCLoRAEngine(lora_engine=lora_engine)

    # Verify injection worked
    assert causal_engine.lora_engine is lora_engine
    assert causal_engine.causal_threshold == 0.1
    assert causal_engine.sample_budget == 1000


def test_identify_causal_paths_empty():
    """Test causal path identification with no modules."""
    mock_model = torch.nn.Linear(10, 2)
    lora_engine = FineTuningEngine(
        task_type=TaskType.SEQ_CLS,
        model=mock_model,
        rank=4,
        alpha=8,
        dropout=0.1
    )

    causal_engine = CausalMonteCLoRAEngine(lora_engine=lora_engine)

    # Mock empty data loader
    class MockDataLoader:
        def __iter__(self):
            return iter([])

    paths = causal_engine.identify_causal_paths(mock_model, MockDataLoader())
    assert isinstance(paths, list)


def test_allocate_budget_equal():
    """Test budget allocation with equal distribution."""
    mock_model = torch.nn.Linear(10, 2)
    lora_engine = FineTuningEngine(
        task_type=TaskType.SEQ_CLS,
        model=mock_model,
        rank=4,
        alpha=8,
        dropout=0.1
    )

    causal_engine = CausalMonteCLoRAEngine(lora_engine=lora_engine)

    paths = ["module1", "module2", "module3"]
    allocation = causal_engine.allocate_budget(paths, total_budget=100)

    assert sum(allocation.values()) == 100
    assert len(allocation) == 3
    assert all(v > 0 for v in allocation.values())
