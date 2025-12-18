"""Accuracy helpers."""
from __future__ import annotations

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / targets.size(0)))
    return res
