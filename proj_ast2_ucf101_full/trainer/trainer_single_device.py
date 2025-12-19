"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import torch
from functools import partial
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.metrics import topk_accuracy
from utils.distributed_utils import get_device


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def build_dataloaders(cfg):
    train_ds = UCF101Dataset(cfg, split="train")
    val_ds = UCF101Dataset(cfg, split="val")
    print(f"[DEBUG] len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    return train_loader, val_loader


def train_single_device(cfg):
    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    train_loader, val_loader = build_dataloaders(cfg)
    if cfg.training.model_type == "video_audio":
        model = VideoAudioAST(
            img_size=cfg.model.img_size,
            num_frames=cfg.model.num_frames,
            num_classes=cfg.model.num_classes,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            patch_size=cfg.model.patch_size,
            audio_feat_dim=cfg.audio.feat_dim,
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
            num_frames=cfg.model.num_frames,
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

    for epoch in range(cfg.train.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            if epoch == 0 and step == 0:
                logger.info("[DEBUG] train batch video.shape=%s", tuple(x.shape))
            opt.zero_grad()
            with autocast(device_type, enabled=cfg.train.amp):
                logits, info = model(x, return_intermediate=True)
                loss_task = F.cross_entropy(logits, y)
                loss = loss_task + cfg.loss.lambda_AST * info["L_AST"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0:
                acc1, acc5 = topk_accuracy(logits.detach(), y, topk=(1, 5))
                log_stats(logger, {"epoch": epoch, "step": step, "loss": loss.item(), "acc1": acc1.item(), "acc5": acc5.item(), "sparsity_token": info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item()})
        validate(model, val_loader, device, logger, epoch, cfg)


def validate(model: nn.Module, loader: DataLoader, device: torch.device, logger, epoch: int, cfg):
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
