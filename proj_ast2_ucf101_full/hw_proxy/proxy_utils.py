"""Utility helpers for proxy loading and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .layer_proxy_model import LayerProxyModel


def load_proxy(weight_path: str, in_dim: int) -> LayerProxyModel:
    model = LayerProxyModel(in_dim=in_dim)
    path = Path(weight_path)
    if path.is_file():
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    return model


def mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs() / (target.abs() + 1e-6)
