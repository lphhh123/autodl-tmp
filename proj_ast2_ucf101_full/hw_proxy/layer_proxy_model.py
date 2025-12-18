"""Generic MLP proxy (SPEC §6.2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerProxyModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, out_dim: int = 1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
