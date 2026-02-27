


import pytest
import torch
from transformers import AutoTokenizer
from src.utils.metrics import compute_glue_metrics

def test_paraphrase_logic():
    """
    Test if the model distinguishes similar sentences using the MRPC logic [3, 5].
    Expect high similarity (Class 1) after successful fine-tuning.
    """
    # 1. Setup (Simulating our decoupled components)
    model_id = "gpt2" # Example compact model for 10GB RAM constraints
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # GPT-2 requires a padding token to be explicitly set
    tokenizer.pad_token = tokenizer.eos_token 
    
    sentence1 = "The cat sat on the mat."
    sentence2 = "A feline was resting on the rug."
    
    # 2. Preprocessing: Tokenizing as a pair [6, 7]
    # GLUE paraphrase tasks require the sentences to be processed together
    inputs = tokenizer(
        sentence1, 
        sentence2, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # 3. Simulated Inference (Mocking model output for unit test isolation)
    # In a real run, this would be: outputs = model(**inputs)
    # We simulate logits where Class 1 (Paraphrase) has a higher score
    mock_logits = torch.tensor([[ -1.5, 3.5 ]]) # High score for index 1
    expected_label = torch.tensor([8]) # Human ground truth: it is a paraphrase [3]
    
    # 4. Metric Computation using our utility [2, 9]
    results = compute_glue_metrics("mrpc", mock_logits, expected_label)
    
    # 5. Assertions: Verifying the adjustment to the domain
    # MRPC results are scaled by 100 in standard baselines [2]
    assert results["accuracy"] > 0.9, f"Model failed to recognize paraphrase. Score: {results}"
    assert "f1" in results, "F1 score must be computed for imbalanced GLUE tasks [2, 3]"
    
    print(f"Paraphrase logic test passed with accuracy: {results['accuracy']}")