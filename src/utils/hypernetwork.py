"""Hypernetwork utilities for Bayesian causal LoRA sampling."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRAHypernetwork(nn.Module):
    """Lightweight PG-Pos hypernetwork for LoRA mean generation."""

    def __init__(
        self,
        n_layers: int,
        out_dim: int,
        d_embed: int = 16,
        max_layers: int = 64,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.out_dim = out_dim

        n_emb = max(n_layers, max_layers)
        self.layer_embeddings = nn.Embedding(n_emb, d_embed)
        self.pos_embeddings = nn.Embedding(max_layers, d_embed)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_embed, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def generate(self, layer_idx: int, pos_idx: int) -> torch.Tensor:
        """Generate a flat mean vector for a LoRA component."""
        layer_idx_t = torch.tensor(layer_idx, dtype=torch.long)
        pos_idx_t = torch.tensor(pos_idx, dtype=torch.long)
        layer_emb = self.layer_embeddings(layer_idx_t)
        pos_emb = self.pos_embeddings(pos_idx_t)
        features = torch.cat([layer_emb, pos_emb], dim=0)
        return self.mlp(features)
