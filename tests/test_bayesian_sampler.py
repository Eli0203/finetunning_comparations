"""Unit tests for BayesianCausalSampler (US2)."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from src.utils.bayesian_sampler import BayesianCausalSampler


class TinyLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.lora_A = nn.Parameter(torch.randn(4, 8))
        self.lora_B = nn.Parameter(torch.randn(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _mock_engine() -> object:
    return SimpleNamespace(
        budget_allocation={"lora_A": 60, "lora_B": 40},
        lora_engine=SimpleNamespace(lora_rank=4),
    )


def test_mog_sampling_shapes_and_cpu_tensors() -> None:
    model = TinyLoRAModel()
    sampler = BayesianCausalSampler(_mock_engine(), model, n_components=4)

    sampled = sampler.sample_batch()

    assert "lora_A" in sampled
    assert "lora_B" in sampled
    assert sampled["lora_A"].shape == model.lora_A.shape
    assert sampled["lora_B"].shape == model.lora_B.shape
    assert sampled["lora_A"].device.type == "cpu"
    assert sampled["lora_B"].device.type == "cpu"


def test_dirichlet_initialization_properties() -> None:
    model = TinyLoRAModel()
    sampler = BayesianCausalSampler(
        _mock_engine(),
        model,
        n_components=4,
        random_dirichlet_init=True,
    )

    components = sampler._ensure_components("lora_A", model.lora_A.numel())

    assert components.pi.shape == (4,)
    assert torch.all(components.pi > 0)
    assert torch.isclose(components.pi.sum(), torch.tensor(1.0), atol=1e-5)


def test_wishart_precision_shape_and_finite_values() -> None:
    model = TinyLoRAModel()
    sampler = BayesianCausalSampler(_mock_engine(), model, n_components=3)

    components = sampler._ensure_components("lora_A", model.lora_A.numel())

    assert components.precision.shape == (3, 4, 4)
    assert torch.isfinite(components.precision).all()


def test_pg_pos_hypernetwork_mean_generation() -> None:
    model = TinyLoRAModel()
    sampler = BayesianCausalSampler(
        _mock_engine(),
        model,
        enable_pg_pos=True,
        n_components=4,
    )

    components = sampler._ensure_components("lora_A", model.lora_A.numel())

    assert sampler._hypernetwork is not None
    assert components.mu.shape == (4, model.lora_A.numel())
    assert torch.any(components.mu != 0)


def test_kfac_block_update_and_reuse() -> None:
    model = TinyLoRAModel()
    sampler = BayesianCausalSampler(_mock_engine(), model, kfac_correlation=True)

    a1 = torch.eye(4)
    g1 = torch.eye(4) * 2
    sampler.update_kfac_blocks("lora_A", a1, g1)
    assert "lora_A" in sampler._kfac_blocks
    assert sampler.get_config()["num_kfac_blocks"] == 1

    a2 = torch.eye(4) * 3
    g2 = torch.eye(4) * 4
    sampler.update_kfac_blocks("lora_A", a2, g2)
    stored_a, stored_g = sampler._kfac_blocks["lora_A"]
    assert torch.allclose(stored_a, a2)
    assert torch.allclose(stored_g, g2)
