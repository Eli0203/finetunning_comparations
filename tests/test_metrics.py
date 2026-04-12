import torch
import pytest

from src.utils.metrics import natural_indirect_effect
from src.utils.metrics import UnifiedEvaluator


def test_natural_indirect_effect_simple():
    # Setup a simple mediator distribution and outcome model.
    # Let P(Y|X=1,M) = [0.2, 0.8] for M=[0,1]
    p_y_given_x_m = torch.tensor([[0.2, 0.8]])

    # Let mediator distributions under X=0 and X=1
    p_m_x = torch.tensor([0.6, 0.4])
    p_m_x_prime = torch.tensor([0.2, 0.8])

    # NIE = E[Y_{x,M_{x'}}] - E[Y_{x,M_x}]
    # E[Y_{x,M_{x'}}] = 0.2*0.2 + 0.8*0.8 = 0.68
    # E[Y_{x,M_x}] = 0.2*0.6 + 0.8*0.4 = 0.44
    expected_nie = 0.68 - 0.44

    nie = natural_indirect_effect(p_y_given_x_m, p_m_x, p_m_x_prime, m_dim=1)
    assert torch.allclose(nie, torch.tensor([expected_nie]), atol=1e-6)


class _StubGlueMetric:
    def compute(self, predictions, references):
        return {}


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr("evaluate.load", lambda *_args, **_kwargs: _StubGlueMetric())
    return UnifiedEvaluator("mrpc")


def _sample_logits_labels():
    logits = torch.tensor([
        [4.0, 1.0],
        [0.5, 1.5],
        [2.0, 0.1],
        [0.2, 2.2],
    ])
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return logits, labels


def test_complete_schema_for_lora(evaluator):
    """T080: lora output must include complete schema with laplace keys as None."""
    logits, labels = _sample_logits_labels()
    result = evaluator.compute_all(logits, labels, method="lora")

    expected_keys = {
        "accuracy",
        "f1",
        "nll",
        "ece",
        "posterior_mean_nll",
        "posterior_variance_mean",
        "epistemic_uncertainty_mean",
    }

    assert set(result.keys()) == expected_keys
    assert result["posterior_mean_nll"] is None
    assert result["posterior_variance_mean"] is None
    assert result["epistemic_uncertainty_mean"] is None


def test_complete_schema_for_laplace_lora(evaluator):
    """T081: laplace_lora output must include complete schema with populated laplace keys."""
    logits, labels = _sample_logits_labels()
    result = evaluator.compute_all(logits, labels, method="laplace_lora")

    expected_keys = {
        "accuracy",
        "f1",
        "nll",
        "ece",
        "posterior_mean_nll",
        "posterior_variance_mean",
        "epistemic_uncertainty_mean",
    }

    assert set(result.keys()) == expected_keys
    assert result["posterior_mean_nll"] is not None
    assert result["posterior_variance_mean"] is not None
    assert result["epistemic_uncertainty_mean"] is not None


def test_complete_schema_for_causal_lora(evaluator):
    """T082: causal_lora output must include complete schema with laplace keys as None."""
    logits, labels = _sample_logits_labels()
    result = evaluator.compute_all(logits, labels, method="causal_lora")

    expected_keys = {
        "accuracy",
        "f1",
        "nll",
        "ece",
        "posterior_mean_nll",
        "posterior_variance_mean",
        "epistemic_uncertainty_mean",
    }

    assert set(result.keys()) == expected_keys
    assert result["posterior_mean_nll"] is None
    assert result["posterior_variance_mean"] is None
    assert result["epistemic_uncertainty_mean"] is None


def test_identical_key_set_across_methods(evaluator):
    """T083: key set must be stable across lora/laplace_lora/causal_lora."""
    logits, labels = _sample_logits_labels()

    lora_keys = set(evaluator.compute_all(logits, labels, method="lora").keys())
    laplace_keys = set(evaluator.compute_all(logits, labels, method="laplace_lora").keys())
    causal_keys = set(evaluator.compute_all(logits, labels, method="causal_lora").keys())

    assert lora_keys == laplace_keys == causal_keys


def test_invalid_method_fails_with_explicit_error(evaluator):
    """T087: method validation must fail explicitly for unsupported methods."""
    logits, labels = _sample_logits_labels()

    with pytest.raises(ValueError, match="unsupported_method|Unsupported method|diagnostic"):
        evaluator.compute_all(logits, labels, method="unknown_method")


def test_invalid_input_shape_fails_with_explicit_error(evaluator):
    """T087: input-shape validation must fail explicitly for mismatched dimensions."""
    bad_logits = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([0, 1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="invalid_shape|shape|diagnostic"):
        evaluator.compute_all(bad_logits, labels, method="lora")


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("method", ["lora", "laplace_lora", "causal_lora"])
def test_schema_stability_across_batch_sizes_and_methods(evaluator, batch_size, method):
    """T088: schema keyset remains stable across methods and batch sizes."""
    logits = torch.randn(batch_size, 3)
    labels = torch.randint(0, 3, (batch_size,), dtype=torch.long)

    result = evaluator.compute_all(logits, labels, method=method)

    expected_keys = {
        "accuracy",
        "f1",
        "nll",
        "ece",
        "posterior_mean_nll",
        "posterior_variance_mean",
        "epistemic_uncertainty_mean",
    }
    assert set(result.keys()) == expected_keys


def test_legacy_core_metric_parity_where_applicable(evaluator):
    """T089: core metrics remain method-agnostic for identical logits/labels."""
    logits, labels = _sample_logits_labels()

    lora = evaluator.compute_all(logits, labels, method="lora")
    laplace = evaluator.compute_all(logits, labels, method="laplace_lora")
    causal = evaluator.compute_all(logits, labels, method="causal_lora")

    for key in ("accuracy", "f1", "nll", "ece"):
        assert lora[key] == pytest.approx(laplace[key], rel=1e-9, abs=1e-9)
        assert lora[key] == pytest.approx(causal[key], rel=1e-9, abs=1e-9)
