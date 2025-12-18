"""Lightweight FLOPs/bytes estimators used by AST and proxy modules."""
from __future__ import annotations

from typing import Dict

import torch


def estimate_vit_flops(seq_len: int, embed_dim: int, num_heads: int, mlp_ratio: float, depth: int) -> Dict[str, torch.Tensor]:
    attn = 2 * seq_len * seq_len * embed_dim * depth / num_heads
    mlp = 2 * seq_len * embed_dim * embed_dim * mlp_ratio * depth
    total = attn + mlp
    return {"attn": torch.tensor(attn), "mlp": torch.tensor(mlp), "total": torch.tensor(total)}
