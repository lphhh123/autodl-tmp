from typing import Dict, Any

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model_video_vit import VideoViT
from . import entropy_utils
from hw_proxy.layer_proxy import LayerHwProxy


def build_model(cfg) -> VideoViT:
    mcfg = cfg.model
    model = VideoViT(
        img_size=mcfg.img_size,
        num_frames=mcfg.num_frames,
        patch_size=mcfg.patch_size,
        in_chans=mcfg.in_chans,
        num_classes=mcfg.num_classes,
        embed_dim=mcfg.embed_dim,
        depth=mcfg.depth,
        num_heads=mcfg.num_heads,
        mlp_ratio=mcfg.mlp_ratio,
        drop_rate=mcfg.get("drop_rate", 0.0),
        attn_drop_rate=mcfg.get("attn_drop_rate", 0.0),
    )
    return model


def build_hw_proxy(cfg) -> LayerHwProxy:
    pcfg = cfg.proxy
    proxy = LayerHwProxy(
        gpu_yaml_path=pcfg.gpu_yaml,
        proxy_weight_dir=pcfg.weight_dir,
        device=pcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return proxy


def compute_hw_loss(
    model: nn.Module,
    proxy: LayerHwProxy,
    cfg,
    input_shape,
) -> torch.Tensor:
    loss_cfg = cfg.loss
    proxy_cfg = cfg.proxy

    alpha_t = loss_cfg.alpha_t
    alpha_m = loss_cfg.alpha_m
    alpha_e = loss_cfg.alpha_e

    latency_norm = loss_cfg.get("latency_norm", 1.0)
    mem_norm = loss_cfg.get("mem_norm", 1.0)
    energy_norm = loss_cfg.get("energy_norm", 1.0)

    device_name = proxy_cfg.device_name

    layer_metas = model.get_layer_metas(input_shape, chip_name=device_name)
    if len(layer_metas) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    hw_cost = 0.0
    for meta in layer_metas:
        pred = proxy.predict_layer(device_name, meta)
        hw_cost = hw_cost + (
            alpha_t * pred["ms"] / latency_norm
            + alpha_m * pred["mem"] / mem_norm
            + alpha_e * pred["energy"] / energy_norm
        )

    hw_loss = torch.tensor(hw_cost, device=next(model.parameters()).device)
    return hw_loss


def train_one_epoch(
    model: nn.Module,
    proxy: LayerHwProxy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    device = next(model.parameters()).device

    loss_cfg = cfg.loss
    proxy_cfg = cfg.proxy

    running_cls = 0.0
    running_sparse = 0.0
    running_hw = 0.0
    running_total = 0.0
    n_samples = 0

    for i, batch in enumerate(dataloader):
        video, label = batch  # video: [B, T, C, H, W]; label: [B]
        video = video.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(video)
        cls_loss = F.cross_entropy(logits, label)

        sparse_loss = entropy_utils.compute_entropy_sparsity_loss(
            model=model,
            lambda_keep=1.0,
            target_start=loss_cfg.get("target_keep_start", 1.0),
            target_end=loss_cfg.get("target_keep_end", 0.5),
        )

        if proxy_cfg.use_hw_loss:
            hw_loss = compute_hw_loss(
                model=model,
                proxy=proxy,
                cfg=cfg,
                input_shape=tuple(video.shape),
            )
        else:
            hw_loss = torch.tensor(0.0, device=device)

        loss = (
            cls_loss
            + loss_cfg.lambda_sparse * sparse_loss
            + loss_cfg.lambda_hw * hw_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        B = video.size(0)
        running_cls += cls_loss.item() * B
        running_sparse += sparse_loss.item() * B
        running_hw += hw_loss.item() * B
        running_total += loss.item() * B
        n_samples += B

        if (i + 1) % cfg.log.print_freq == 0:
            print(
                f"Epoch[{epoch}] Iter[{i+1}/{len(dataloader)}] " \
                f"cls={cls_loss.item():.4f} " \
                f"sparse={sparse_loss.item():.4f} " \
                f"hw={hw_loss.item():.4f} " \
                f"total={loss.item():.4f}",
                flush=True,
            )

    return {
        "cls": running_cls / n_samples,
        "sparse": running_sparse / n_samples,
        "hw": running_hw / n_samples,
        "total": running_total / n_samples,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, cfg) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.inference_mode():
        for video, label in dataloader:
            video = video.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(video)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.numel()
    acc = correct / max(total, 1)
    return acc


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    acc: float,
                    path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "acc": acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
