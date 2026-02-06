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
    aggregate: str = "clip",
) -> float:
    model.eval()
    aggregate = str(aggregate).lower()
    do_video = aggregate in {"video", "video_avg", "video_mean"}

    total = 0
    correct = 0
    video_logits_sum = {}
    video_counts = {}
    video_labels = {}

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

        if do_video:
            vids = batch.get("video_id", None)
            if vids is not None:
                logits_cpu = logits.detach().float().cpu()
                y_cpu = y.detach().cpu()
                for j, vid in enumerate(vids):
                    k = str(vid)
                    if k not in video_logits_sum:
                        video_logits_sum[k] = logits_cpu[j].clone()
                        video_counts[k] = 1
                        video_labels[k] = int(y_cpu[j].item())
                    else:
                        video_logits_sum[k] += logits_cpu[j]
                        video_counts[k] += 1

    acc_clip = float(correct / max(1, total))
    if not do_video or len(video_logits_sum) == 0:
        return acc_clip

    vids_all = list(video_logits_sum.keys())
    logits_video = torch.stack(
        [video_logits_sum[k] / float(max(1, video_counts[k])) for k in vids_all],
        dim=0,
    )
    labels_video = torch.tensor([video_labels[k] for k in vids_all], dtype=torch.long)
    pred_video = logits_video.argmax(dim=1)
    acc_video = float((pred_video == labels_video).float().mean().item())
    return acc_video
