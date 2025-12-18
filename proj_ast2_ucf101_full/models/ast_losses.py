"""Regularizers for AST2.0-lite sparsity."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ASTRegularizer(nn.Module):
    """Sparsity regularization combining token/head/channel gates."""

    def __init__(
        self,
        lambda_token: float = 1.0,
        lambda_head: float = 1.0,
        lambda_channel: float = 1.0,
        target_sparsity: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_token = lambda_token
        self.lambda_head = lambda_head
        self.lambda_channel = lambda_channel
        self.target_sparsity = target_sparsity

    def forward(self, masks_dict: Dict[str, torch.Tensor], flops_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((), device=next(iter(masks_dict.values())).device)
        if "token" in masks_dict:
            token_mask = masks_dict["token"]
            loss = loss + self.lambda_token * (1.0 - token_mask.mean())
        if "head" in masks_dict:
            head_mask = masks_dict["head"]
            loss = loss + self.lambda_head * (1.0 - head_mask.mean())
        if "channel" in masks_dict:
            ch_mask = masks_dict["channel"]
            loss = loss + self.lambda_channel * (1.0 - ch_mask.mean())

        if self.target_sparsity is not None and "token" in masks_dict:
            dens = masks_dict["token"].mean()
            loss = loss + (dens - (1 - self.target_sparsity)) ** 2

        if flops_dict:
            flops_total = torch.stack([v for v in flops_dict.values()]).sum()
            loss = loss + 1e-6 * flops_total
        return loss
