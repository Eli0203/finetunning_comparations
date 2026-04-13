from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/causal_finetuning_demo.ipynb")


def _notebook_text() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_comparison_dataframe_includes_required_columns():
    text = _notebook_text()
    assert 'comparison_df = pd.concat([lora_df, laplace_df, causal_df], ignore_index=True)' in text
    assert "suite" in text
    assert "task" in text
    assert "accuracy" in text
    assert "nll" in text
    assert '\\"LoRA Only\\"' in text
    assert '\\"Laplace-LoRA Only\\"' in text
    assert '\\"Causal-LoRA Only\\"' in text
    assert "runtime_seconds" in text
    assert "output_dir" in text
    assert "model_dir" in text


def test_notebook_contains_accuracy_and_loss_plots():
    text = _notebook_text()
    assert "plot_df" in text
    assert "run_label" in text
    assert "Accuracy Comparison" in text
    assert "Loss Comparison (NLL)" in text
    assert "plt.subplots(1, 2" in text
    assert "plt.show()" in text
