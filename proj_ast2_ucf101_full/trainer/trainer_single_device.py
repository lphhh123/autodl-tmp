"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from functools import partial
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.hw_loss import compute_hw_loss
from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.metrics import topk_accuracy
from utils.distributed_utils import get_device
from utils.eval_utils import eval_acc1
from utils.stable_hw import (
    apply_accuracy_guard,
    init_hw_refs_from_baseline_stats,
    stable_hw_log_fields,
    stable_hw_schedule,
    update_hw_refs_from_stats,
)


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
    hw_proxy = LayerHwProxy(cfg.hw.device_name, cfg.hw.gpu_yaml, cfg.hw.proxy_weight_dir)
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    stable_state: Dict[str, Any] = {}

    best_acc = 0.0
    last_acc = 0.0
    for epoch in range(cfg.train.epochs):
        stable_hw_enabled = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
        if stable_hw_enabled:
            stable_hw_schedule(epoch, stable_hw_cfg, stable_state)
            prev_val = stable_state.get("val_acc1_last", None)
            apply_accuracy_guard(
                epoch=epoch,
                stable_hw_cfg=stable_hw_cfg,
                stable_hw_state=stable_state,
                val_acc1=prev_val,
                train_acc1_ema=float(stable_state.get("train_acc1_ema", 0.0))
                if stable_state.get("train_acc1_ema") is not None
                else None,
            )
        lambda_hw_eff = float(
            stable_state.get(
                "lambda_hw_effective",
                stable_state.get("lambda_hw_after_guard", stable_state.get("lambda_hw", float(getattr(cfg.hw, "lambda_hw", 0.0)))),
            )
        )
        model.train()
        last_hw_stats = None
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
                L_task = F.cross_entropy(logits, y)

                model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                L_hw, hw_stats = compute_hw_loss(
                    cfg,
                    hw_proxy,
                    model_info=model_info,
                    stable_hw_cfg=stable_hw_cfg,
                    stable_hw_state=stable_state,
                )
                if stable_hw_cfg:
                    last_hw_stats = {k: float(v) for k, v in hw_stats.items()}

                L_ast = info["L_AST"] if isinstance(info, dict) and "L_AST" in info else logits.new_tensor(0.0)

                loss = L_task + float(getattr(cfg.loss, "lambda_AST", 1.0)) * L_ast + lambda_hw_eff * L_hw
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
                    "lambda_hw": float(lambda_hw_eff),
                    "hw_loss": float(L_hw),
                }
                stats.update(
                    {
                        "total_latency_ms": float(hw_stats.get("latency_ms", 0.0)),
                        "peak_mem_mb": float(hw_stats.get("mem_mb", 0.0)),
                        "comm_ms": float(hw_stats.get("comm_ms", 0.0)),
                    }
                )
                log_stats(logger, stats)
        last_acc = validate(model, val_loader, device, logger, epoch, cfg)
        best_acc = max(best_acc, last_acc)
        if stable_hw_enabled:
            apply_accuracy_guard(
                epoch=epoch,
                stable_hw_cfg=stable_hw_cfg,
                stable_hw_state=stable_state,
                val_acc1=float(last_acc) if last_acc is not None else None,
                train_acc1_ema=float(stable_state.get("train_acc1_ema", 0.0))
                if stable_state.get("train_acc1_ema") is not None
                else None,
            )

            # ---- v5.4: update HW ref via EMA if not from dense baseline ----
            init_hw_refs_from_baseline_stats(stable_hw_cfg, stable_state)
            if stable_state.get("hw_ref_source", "") != "dense_baseline":
                update_hw_refs_from_stats(stable_hw_cfg, stable_state, last_hw_stats or {})
        if metrics_path:
            metrics = {
                "epoch": int(epoch),
                "acc1": float(last_acc),
                "best_acc1": float(best_acc),
                "loss": float(loss.item()),
                "sparsity_token": float(info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item()),
                "rho_target": float(getattr(cfg.ast, "rho_target", 0.0)),
                "lambda_hw": float(lambda_hw_eff),
                "hw_loss": float(L_hw),
                "stable_hw_disabled": not bool(getattr(cfg.stable_hw, "enabled", False))
                if getattr(cfg, "stable_hw", None)
                else True,
            }
            metrics["stable_hw"] = stable_hw_log_fields(stable_state)
            metrics.update({k: float(v) for k, v in hw_stats.items()})
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        if out_dir is not None and stable_hw_cfg:
            with (out_dir / "stable_hw_state.json").open("w", encoding="utf-8") as f:
                json.dump(stable_state, f, indent=2)


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
