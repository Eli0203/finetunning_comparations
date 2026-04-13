import json

import pytest

import main as runtime_main


class _DummySettings:
    experiment_type = "lora"
    task_name = "mrpc"


def test_main_reraises_structured_failure_by_default(monkeypatch):
    def _raise_settings(*args, **kwargs):
        raise ValueError("MPTY_BUDGET_ALLOCATION")

    monkeypatch.delenv("MAIN_FAILURE_MODE", raising=False)
    monkeypatch.setattr(runtime_main.SettingsFactory, "create_settings", _raise_settings)

    with pytest.raises(RuntimeError) as exc_info:
        runtime_main.main()

    payload = json.loads(str(exc_info.value))
    assert payload["status"] == "failure"
    assert payload["error_type"] == "ValueError"
    assert "MPTY_BUDGET_ALLOCATION" in payload["error_message"]


def test_main_returns_structured_failure_when_configured(monkeypatch):
    def _raise_settings(*args, **kwargs):
        raise RuntimeError("fatal")

    monkeypatch.setenv("MAIN_FAILURE_MODE", "return")
    monkeypatch.setattr(runtime_main.SettingsFactory, "create_settings", _raise_settings)

    result = runtime_main.main()

    assert result["status"] == "failure"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "fatal"
    assert result["experiment_type"] is None


def test_persist_adapter_artifacts_uses_supported_apis_only():
    class _DummyTrainer:
        def __init__(self):
            self.saved_to = None

        def save_model(self, output_dir):
            self.saved_to = output_dir

    class _DummyModel:
        def __init__(self):
            self.saved_to = None

        def save_pretrained(self, output_dir):
            self.saved_to = output_dir

    trainer = _DummyTrainer()
    model = _DummyModel()

    runtime_main._persist_adapter_artifacts(
        trainer=trainer,
        model=model,
        output_dir="output/test_adapter_persistence",
    )

    assert trainer.saved_to == "output/test_adapter_persistence"
    assert model.saved_to == "output/test_adapter_persistence"
