
"""Dummy AST2.0-lite training script.

This trains VideoViT on random data, with:
  - classification loss (cross-entropy)
  - sparsity loss over keep logits
  - a simple hardware loss using single-device LayerHwProxy

It does *not* really prune tokens; it just drives keep logits and
hardware cost to ensure the end-to-end pipeline (model + hw proxy)
works correctly without a real dataset.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import load_config
from data.dummy_video_dataset import DummyVideoDataset
from ast2.model_video_vit import VideoViT
from ast2.entropy_utils import sparsity_regularizer
from hw_proxy.layer_proxy import LayerHwProxy


def compute_hw_loss(
    model: VideoViT,
    proxy: LayerHwProxy,
    lambda_hw: float = 1e-3,
) -> torch.Tensor:
    layer_metas = model.get_layer_metas()
    if not layer_metas:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    total_ms = 0.0
    for row in layer_metas:
        pred = proxy.predict(row)
        total_ms += pred["ms"]
    hw_loss = lambda_hw * (total_ms / 100.0)  # scale a bit
    return torch.tensor(hw_loss, device=next(model.parameters()).device)


def main():
    cfg = load_config("configs/ast2_dummy.yaml")

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    # Model
    model = VideoViT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.model.num_frames,
        in_chans=cfg.model.in_chans,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    ).to(device)

    # Dummy dataset
    dataset = DummyVideoDataset(
        num_samples=32,
        num_classes=cfg.model.num_classes,
        in_chans=cfg.model.in_chans,
        num_frames=cfg.model.num_frames,
        img_size=cfg.model.img_size,
    )
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=True)

    # HW proxy
    proxy = LayerHwProxy(
        device_name=str(cfg.hw.device_name),
        gpu_yaml=str(cfg.hw.gpu_yaml),
        weight_dir="proxy_weights",
    )

    criterion = nn.CrossEntropyLoss()
    lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    print("==== Start dummy AST2.0-lite training (random data) ====")
    model.train()
    for epoch in range(int(cfg.train.epochs)):
        for it, (video, label) in enumerate(loader):
            video = video.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(video)
            cls_loss = criterion(logits, label)

            keep_logits = model.get_keep_logits()
            sparse_loss = sparsity_regularizer(keep_logits, target_keep=0.7, alpha=float(cfg.loss.lambda_sparse))

            hw_loss = compute_hw_loss(model, proxy, lambda_hw=float(cfg.loss.lambda_hw))

            loss = cls_loss + sparse_loss + hw_loss
            loss.backward()
            optimizer.step()

            if (it + 1) % 2 == 0:
                print(
                    f"Epoch [{epoch}] Iter [{it+1}/{len(loader)}] "
                    f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                    f"sparse={sparse_loss.item():.4f} hw={hw_loss.item():.4f}"
                )

    print("==== Done. AST2.0-lite dummy training finished. ====")


if __name__ == "__main__":
    main()
