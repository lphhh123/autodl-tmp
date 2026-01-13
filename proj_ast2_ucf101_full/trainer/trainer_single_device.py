"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hw_proxy.hw_loss import compute_hw_loss
from hw_proxy.layer_hw_proxy import LayerHwProxy
from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.metrics import topk_accuracy
from utils.distributed_utils import get_device
from utils.lambda_hw import resolve_lambda_hw


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(cfg):
    train_ds = UCF101Dataset(cfg, split="train")
    val_ds = UCF101Dataset(cfg, split="val")
    print(f"[DEBUG] len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
    )
    return train_loader, val_loader


def train_single_device(cfg, out_dir: str | Path | None = None):
    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    metrics_path = None
    if out_dir is None and hasattr(cfg, "train") and getattr(cfg.train, "out_dir", None):
        out_dir = Path(cfg.train.out_dir)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
    train_loader, val_loader = build_dataloaders(cfg)
    model_type = getattr(cfg.training, "model_type", "video")
    num_frames = int(getattr(cfg.data, "num_frames", cfg.model.num_frames))
    audio_feat_dim = int(getattr(cfg.data, "audio_feat_dim", cfg.audio.feat_dim))
    if model_type == "video_audio":
        model = VideoAudioAST(
            img_size=cfg.model.img_size,
            num_frames=num_frames,
            num_classes=cfg.model.num_classes,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            patch_size=cfg.model.patch_size,
            audio_feat_dim=audio_feat_dim,
            in_chans=cfg.model.in_chans,
            drop_rate=cfg.model.drop_rate,
            attn_drop=cfg.model.attn_drop,
            drop_path_rate=cfg.model.drop_path_rate,
            use_ast_prune=cfg.ast.use_ast_prune,
            ast_cfg=cfg.ast,
        ).to(device)
    else:
        model = VideoViT(
            img_size=cfg.model.img_size,
            num_frames=num_frames,
            num_classes=cfg.model.num_classes,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            patch_size=cfg.model.patch_size,
            in_chans=cfg.model.in_chans,
            drop_rate=cfg.model.drop_rate,
            attn_drop=cfg.model.attn_drop,
            drop_path_rate=cfg.model.drop_path_rate,
            use_ast_prune=cfg.ast.use_ast_prune,
            ast_cfg=cfg.ast,
        ).to(device)

    lr = _as_float(cfg.train.lr, "cfg.train.lr")
    weight_decay = _as_float(cfg.train.weight_decay, "cfg.train.weight_decay")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=cfg.train.amp)
    hw_proxy = None
    if resolve_lambda_hw(cfg, stable_hw_state=None) > 0:
        hw_cfg = getattr(cfg, "hw", None)
        if hw_cfg is None:
            raise ValueError("cfg.hw is required when hw.lambda_hw > 0")
        hw_proxy = LayerHwProxy(hw_cfg.device_name, hw_cfg.gpu_yaml, hw_cfg.proxy_weight_dir)

    best_acc = 0.0
    last_acc = 0.0
    for epoch in range(cfg.train.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            if epoch == 0 and step == 0:
                logger.info("[DEBUG] train batch video.shape=%s", tuple(x.shape))
            opt.zero_grad()
            with autocast(device_type, enabled=cfg.train.amp):
                if model_type == "video_audio":
                    logits, info = model(x, batch["audio"].to(device), return_intermediate=True)
                else:
                    logits, info = model(x, return_intermediate=True)
                loss_task = F.cross_entropy(logits, y)
                lambda_hw = resolve_lambda_hw(cfg, stable_hw_state=None)
                hw_loss = 0.0
                hw_metrics = {}
                if lambda_hw > 0 and hw_proxy is not None:
                    model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                    hw_loss, hw_metrics = compute_hw_loss(
                        cfg,
                        hw_proxy,
                        model_info=model_info,
                        stable_hw_cfg=getattr(cfg, "stable_hw", None),
                        stable_hw_state=None,
                    )
                loss = loss_task + cfg.loss.lambda_AST * info["L_AST"] + float(lambda_hw) * float(hw_loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0:
                acc1, acc5 = topk_accuracy(logits.detach(), y, topk=(1, 5))
                stats = {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "acc1": acc1.item(),
                    "acc5": acc5.item(),
                    "sparsity_token": info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item(),
                    "lambda_hw": float(lambda_hw),
                    "hw_loss": float(hw_loss),
                }
                stats.update(hw_metrics)
                log_stats(logger, stats)
        last_acc = validate(model, val_loader, device, logger, epoch, cfg)
        best_acc = max(best_acc, last_acc)
        if metrics_path:
            metrics = {
                "epoch": int(epoch),
                "acc1": float(last_acc),
                "best_acc1": float(best_acc),
                "loss": float(loss.item()),
                "sparsity_token": float(info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item()),
                "rho_target": float(getattr(cfg.ast, "rho_target", 0.0)),
                "lambda_hw": float(lambda_hw),
                "hw_loss": float(hw_loss),
            }
            metrics.update({k: float(v) for k, v in hw_metrics.items()})
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)


def validate(model: nn.Module, loader: DataLoader, device: torch.device, logger, epoch: int, cfg) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            if cfg.training.model_type == "video_audio":
                logits = model(x, batch["audio"].to(device))
            else:
                logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = correct / max(1, total)
    logger.info(f"[val] epoch {epoch} acc={acc:.4f}")
    return float(acc)
