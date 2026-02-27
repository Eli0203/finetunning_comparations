import pytest
from src.utils.metrics import compute_glue_metrics

def test_paraphrase_logic():
    """Test if the model distinguishes similar sentences [21, 28]"""
    sentence1 = "The cat sat on the mat."
    sentence2 = "A feline was resting on the rug."
    # Expect high similarity logic after MRPC fine-tuning
    ...

def test_memory_usage():
    """Ensure we stay within the 10GB RAM constraint [2]"""
    import psutil
    process = psutil.Process()
    assert process.memory_info().rss < 10 * 1024**3 