from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def eval_acc1(
    model: nn.Module,
    loader,
    device: torch.device,
    model_type: str = "video",
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total = 0
    correct = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        x = batch["video"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if model_type == "video_audio":
            audio = batch["audio"].to(device, non_blocking=True)
            logits = model(x, audio)
        else:
            logits = model(x)

        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())

    return float(correct / max(1, total))
