import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

from utils.config import load_config
from ast2.trainer_ast2_single import (
    build_model,
    build_hw_proxy,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)

from data.ucf101_dataset import UCF101VideoDataset


class DummyVideoDataset(Dataset):
    """
    Simple synthetic dataset used only as a fallback when you don't have
    real videos ready. Each sample is random noise with an integer label.
    """

    def __init__(self, num_samples: int = 1024, num_frames: int = 8, img_size: int = 224, num_classes: int = 101):
        super().__init__()
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(3, self.num_frames, self.img_size, self.img_size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def build_dataloaders(cfg):
    """
    Build train/val dataloaders.

    - If cfg.data.dataset == "UCF101", we use the real UCF-101 dataset and
      randomly split it into train/val according to cfg.data.val_ratio
      (e.g., 0.2 gives a 4:1 split for pretraining).
    - Otherwise, we fall back to DummyVideoDataset.
    """
    dataset_name = str(cfg.data.dataset).upper()

    if dataset_name == "UCF101":
        dataset = UCF101VideoDataset(
            root=cfg.data.root,
            clip_len=cfg.data.clip_len,
            img_size=cfg.model.img_size,
            frame_sample_strategy=getattr(cfg.data, "frame_sample_strategy", "uniform"),
        )

        val_ratio = float(getattr(cfg.data, "val_ratio", 0.2))
        n_total = len(dataset)
        n_val = max(1, int(n_total * val_ratio))
        n_train = max(1, n_total - n_val)

        generator = torch.Generator().manual_seed(getattr(cfg.train, "seed", 42))
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    else:
        print("[run_ast2_single] cfg.data.dataset != 'UCF101', use DummyVideoDataset instead.")
        num_samples = 1024
        dataset = DummyVideoDataset(
            num_samples=num_samples,
            num_frames=cfg.model.num_frames,
            img_size=cfg.model.img_size,
            num_classes=cfg.model.num_classes,
        )
        # 8:2 split
        n_val = max(1, num_samples // 5)
        n_train = num_samples - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="AST2.0 single-GPU training entry (UCF-101).")
    parser.add_argument("--config", type=str, default="configs/ast2_single_gpu.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.log.out_dir, exist_ok=True)

    device = torch.device(cfg.train.device)

    # 1) model & hardware proxy
    model = build_model(cfg).to(device)
    hw_proxy = build_hw_proxy(cfg)

    # 2) data
    train_loader, val_loader = build_dataloaders(cfg)

    # 3) optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.train.epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.train.epochs} =====")
        train_one_epoch(
            model=model,
            hw_proxy=hw_proxy,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            hw_proxy=hw_proxy,
            val_loader=val_loader,
            epoch=epoch,
            cfg=cfg,
            device=device,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val_loss,
            cfg=cfg,
            is_best=is_best,
        )

    print("\n[run_ast2_single] Training finished. Best val loss = {:.6f}".format(best_val_loss))


if __name__ == "__main__":
    main()
