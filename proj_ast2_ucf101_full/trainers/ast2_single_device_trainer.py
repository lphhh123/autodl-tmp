"""Single-device AST2.0-lite trainer."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets.ucf101_dataset import UCF101Dataset, collate_video_batch
from models.ast_losses import ASTRegularizer
from models.vit_video import VideoViT
from utils.distributed_utils import get_device
from utils.logging_utils import log_stats, setup_logger
from utils.metric_acc import topk_accuracy


def train_ast2_single_device(cfg):
    device = get_device(cfg.train.device)
    logger = setup_logger(cfg.train.output_dir)

    train_ds = UCF101Dataset(cfg.data.root, cfg.data.split_train, clip_len=cfg.data.clip_len, img_size=cfg.data.img_size, is_train=True)
    val_ds = UCF101Dataset(cfg.data.root, cfg.data.split_val, clip_len=cfg.data.clip_len, img_size=cfg.data.img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=collate_video_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=collate_video_batch)

    model_kwargs = dict(
        img_size=cfg.model.img_size,
        num_frames=cfg.model.num_frames,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        use_ast2=cfg.model.use_ast2,
        ast=getattr(cfg.model, "ast", {}),
    )
    model = VideoViT(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    ast_reg = ASTRegularizer(
        lambda_token=cfg.ast_loss.lambda_token,
        lambda_head=cfg.ast_loss.lambda_head,
        lambda_channel=cfg.ast_loss.lambda_channel,
        target_sparsity=cfg.ast_loss.target_sparsity,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=getattr(cfg.train, "use_amp", True))

    for epoch in range(cfg.train.epochs):
        model.train()
        for step, (video, target) in enumerate(train_loader):
            video = video.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            with autocast(enabled=getattr(cfg.train, "use_amp", True)):
                logits, masks_dict, flops_dict = model(video)
                loss_cls = criterion(logits, target)
                loss_ast = ast_reg(masks_dict, flops_dict)
                loss = loss_cls + loss_ast
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % cfg.train.log_interval == 0:
                acc1, acc5 = topk_accuracy(logits.detach(), target, topk=(1, 5))
                log_stats(logger, step + epoch * len(train_loader), {"loss": loss.item(), "acc1": acc1.item(), "acc5": acc5.item()})

        validate(model, val_loader, criterion, device, logger, epoch)


def validate(model, loader, criterion, device, logger, epoch):
    model.eval()
    loss_total = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for video, target in loader:
            video = video.to(device)
            target = target.to(device)
            logits, _, _ = model(video)
            loss_total += criterion(logits, target).item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = correct / max(1, total)
    logger.info(f"[val] epoch {epoch} loss={loss_total / max(1, len(loader)):.4f} acc={acc:.4f}")
