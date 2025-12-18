"""Multi-granularity gating modules for AST2.0-lite."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGate(nn.Module):
    """Token-level gate driven by entropy and region weights."""

    def __init__(self, embed_dim: int, init_keep_ratio: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.score = nn.Linear(embed_dim, 1)
        self.init_keep_ratio = init_keep_ratio

    def forward(
        self,
        x: torch.Tensor,
        entropy_t: torch.Tensor,
        entropy_s: torch.Tensor,
        region_ids: torch.Tensor,
        region_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply token gating.

        Parameters
        ----------
        x: torch.Tensor
            Token tensor ``[B, N, C]``.
        entropy_t: torch.Tensor
            Temporal entropy ``[B, N]``.
        entropy_s: torch.Tensor
            Spatial entropy ``[B, N]``.
        region_ids: torch.Tensor
            Region assignment ``[N]``.
        region_weights: torch.Tensor
            Region weights ``[B, R]``.
        """
        base_score = self.score(x).squeeze(-1)  # [B, N]
        reg_score = region_weights[:, region_ids]
        combined = base_score + entropy_t + entropy_s + reg_score
        keep_ratio = torch.sigmoid(combined).mean().detach()
        target_keep = self.init_keep_ratio
        threshold = torch.quantile(combined, 1 - target_keep)
        mask = (combined >= threshold).float().unsqueeze(-1)  # [B, N, 1]
        gated = x * mask
        return gated, mask


class HeadGate(nn.Module):
    """Head-level gate for multi-head attention."""

    def __init__(self, num_heads: int, init_keep_ratio: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.logits = nn.Parameter(torch.zeros(num_heads))
        self.init_keep_ratio = init_keep_ratio

    def forward(self, attn_weights: torch.Tensor, entropy_head: torch.Tensor | None = None) -> torch.Tensor:
        head_score = torch.sigmoid(self.logits)
        if entropy_head is not None:
            head_score = head_score * (1.0 + entropy_head)
        keep = head_score
        threshold = torch.quantile(keep, 1 - self.init_keep_ratio)
        mask = (keep >= threshold).float()
        return mask


class ChannelGate(nn.Module):
    """Channel-wise squeeze-excitation style gate."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            pooled = x.mean(dim=1)
        else:
            pooled = x.mean(dim=list(range(2, x.dim())))
        mask = self.net(pooled)
        return mask


class BlockGate(nn.Module):
    """Optional gate toggling entire blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, block_stats: dict | None = None) -> torch.Tensor:
        prob = torch.sigmoid(self.logit)
        if block_stats is not None and "entropy" in block_stats:
            prob = prob * (1.0 + block_stats["entropy"])
        return prob
