"""Entropy utilities for AST2.0-lite."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_temporal_entropy(tokens: torch.Tensor) -> torch.Tensor:
    """Compute temporal entropy across frames.

    Parameters
    ----------
    tokens: torch.Tensor
        Token tensor shaped ``[B, T, N, C]`` or ``[B, N, T, C]``.

    Returns
    -------
    torch.Tensor
        Entropy estimate per token: ``[B, T, N]``.
    """
    if tokens.dim() != 4:
        raise ValueError("tokens must have shape [B, T, N, C] or [B, N, T, C]")
    if tokens.shape[1] != tokens.shape[2]:
        # assume [B, T, N, C]
        x = tokens
    else:
        # ambiguous, default to [B, T, N, C]
        x = tokens

    b, t, n, c = x.shape
    diff = x[:, 1:, :, :] - x[:, :-1, :, :]
    var = diff.pow(2).mean(dim=1)
    entropy = torch.log1p(var.mean(dim=-1))  # [B, N]
    entropy = entropy.unsqueeze(1).expand(-1, t, -1)
    return entropy


def compute_spatial_entropy(features: torch.Tensor) -> torch.Tensor:
    """Compute spatial entropy for patches.

    Parameters
    ----------
    features: torch.Tensor
        Feature tensor shaped ``[B, T, H, W, C]`` or ``[B, T, N, C]``.

    Returns
    -------
    torch.Tensor
        Spatial entropy per frame: ``[B, T, N]``.
    """
    if features.dim() == 5:
        b, t, h, w, c = features.shape
        feats = features.view(b, t, h * w, c)
    elif features.dim() == 4:
        b, t, n, c = features.shape
        feats = features
    else:
        raise ValueError("features must be [B, T, H, W, C] or [B, T, N, C]")

    mean = feats.mean(dim=-1, keepdim=True)
    centered = feats - mean
    var = (centered ** 2).mean(dim=-1)
    entropy = torch.log1p(var)
    return entropy
