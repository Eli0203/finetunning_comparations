import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.finetuner.laplace_engine import LaplaceLoRAEngine


class TinyLaplaceDataset(Dataset):
    def __init__(self, size: int = 8, seq_len: int = 5, vocab_size: int = 32):
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, 2, (size,), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "label": self.labels[idx],
        }


class TinyLaplaceModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lora_A = nn.Linear(hidden_dim, 3, bias=False)
        self.classifier = nn.Linear(3, 2)

    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        embedded = self.embedding(input_ids).mean(dim=1)
        hidden = self.lora_A(embedded)
        logits = self.classifier(hidden)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return types.SimpleNamespace(loss=loss, logits=logits)


def build_lora_engine_stub():
    model = TinyLaplaceModel()
    stub = types.SimpleNamespace(
        peft_model=model,
        _config=types.SimpleNamespace(r=4),
        lora_rank=4,
    )
    return stub


def test_accumulate_curvature_collects_hook_factors():
    lora_engine = build_lora_engine_stub()
    engine = LaplaceLoRAEngine(lora_engine=lora_engine, prior_precision=1.0, nkfac=4)
    loader = DataLoader(TinyLaplaceDataset(), batch_size=2)

    engine.accumulate_curvature(loader)

    assert "lora_A" in engine.curvature_factors
    activation_factor, gradient_factor = engine.curvature_factors["lora_A"]
    assert activation_factor.shape[0] == lora_engine.peft_model.lora_A.in_features
    assert gradient_factor.shape[0] == lora_engine.peft_model.lora_A.out_features
    assert activation_factor.shape[1] <= engine.nkfac
    assert gradient_factor.shape[1] <= engine.nkfac


def test_get_marginal_likelihood_is_finite_after_curvature_pass():
    lora_engine = build_lora_engine_stub()
    engine = LaplaceLoRAEngine(lora_engine=lora_engine, prior_precision=0.5, nkfac=4)
    loader = DataLoader(TinyLaplaceDataset(), batch_size=2)

    engine.accumulate_curvature(loader)
    evidence = engine.get_marginal_likelihood(log_likelihood=-1.25)

    assert isinstance(evidence, float)
    assert torch.isfinite(torch.tensor(evidence)).item()