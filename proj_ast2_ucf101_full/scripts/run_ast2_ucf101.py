import os
import sys
import math
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils.config import load_config
from ast2.model_video_vit import VideoViT
from ast2 import entropy_utils
from hw_proxy.layer_proxy import LayerHwProxy

# 混合精度
from torch.cuda.amp import autocast, GradScaler


# ------------------------------
#  UCF101 抽帧版数据集
# ------------------------------

class UCF101FramesDataset(Dataset):
    """
    使用抽帧后的 UCF101 数据集:
      - frame_root: 形如 data/ucf101/frames
      - split_file: 官方 trainlist01.txt / testlist01.txt
        每行类似: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1"
      - 实际帧目录: frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/frame_000001.jpg
    """

    def __init__(
        self,
        frame_root: Path,
        split_file: Path,
        label_to_idx: Dict[str, int],
        clip_len: int = 8,
        img_size: int = 224,
        is_train: bool = True,
    ):
        super().__init__()
        self.frame_root = frame_root
        self.clip_len = clip_len
        self.img_size = img_size
        self.is_train = is_train
        self.label_to_idx = label_to_idx

        # 解析 split 文件
        self.samples: List[Tuple[Path, int]] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # trainlist: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1"
                # testlist:  "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
                parts = line.split()
                rel_path = parts[0]  # "ApplyEyeMakeup/xxx.avi"
                cls_name = rel_path.split("/")[0]
                if cls_name not in label_to_idx:
                    # 不应该发生，如果 train/val label_to_idx 是联合构建的
                    continue
                label = label_to_idx[cls_name]

                video_stem = Path(rel_path).stem  # 去掉 .avi
                frames_dir = frame_root / cls_name / video_stem
                if not frames_dir.is_dir():
                    # 可能抽帧过程中有缺失，跳过
                    continue

                self.samples.append((frames_dir, label))

        if len(self.samples) == 0:
            print(f"[WARN] No samples found in split: {split_file}")

        # 图像预处理
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(img_size * 1.14)),
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(img_size * 1.14)),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, frames_dir: Path) -> List[Image.Image]:
        frame_files = sorted(
            [p for p in frames_dir.glob("*.jpg") if p.is_file()]
        )
        if len(frame_files) == 0:
            return []

        # 采样 clip_len 帧
        if len(frame_files) >= self.clip_len:
            # 均匀采样
            indices = torch.linspace(
                0, len(frame_files) - 1, steps=self.clip_len
            ).long()
            files = [frame_files[i.item()] for i in indices]
        else:
            # 帧数太少, 则循环补足
            repeat = math.ceil(self.clip_len / len(frame_files))
            files = (frame_files * repeat)[: self.clip_len]

        imgs = []
        for f in files:
            with Image.open(f) as img:
                img = img.convert("RGB")
                imgs.append(img.copy())
        return imgs

    def __getitem__(self, idx: int):
        frames_dir, label = self.samples[idx]
        imgs = self._load_frames(frames_dir)
        if len(imgs) == 0:
            # 理论上不会走到，兜底返回一个黑图 clip
            img = Image.new("RGB", (self.img_size, self.img_size))
            imgs = [img] * self.clip_len

        # 对每帧做 transform, 然后堆成 [T, C, H, W]
        frames_tensor = []
        for img in imgs:
            frames_tensor.append(self.transform(img))  # [C,H,W]
        video = torch.stack(frames_tensor, dim=0)  # [T, C, H, W]

        return video, label


# ------------------------------
#  构建模型
# ------------------------------

def build_model(cfg, device):
    """
    根据 cfg.model 构建 VideoViT。
    并自动过滤掉 VideoViT.__init__ 不支持的参数（例如 drop_path_rate）。
    """
    model_cfg = cfg.model

    # 先把可能用到的字段都取出来
    raw_kwargs = {
        "img_size": model_cfg.img_size,
        "num_frames": model_cfg.num_frames,
        "num_classes": model_cfg.num_classes,
        "embed_dim": model_cfg.embed_dim,
        "depth": model_cfg.depth,
        "num_heads": model_cfg.num_heads,
        "mlp_ratio": model_cfg.mlp_ratio,
        "patch_size": model_cfg.patch_size,
        "in_chans": model_cfg.in_chans,
        "drop_rate": getattr(model_cfg, "drop_rate", 0.0),
        "attn_drop": getattr(model_cfg, "attn_drop", 0.0),
        # "drop_path_rate": getattr(model_cfg, "drop_path_rate", 0.0),  # 旧参数，先不传
    }

    # 根据 VideoViT.__init__ 的签名过滤一遍，避免传入不支持的 key
    import inspect

    sig = inspect.signature(VideoViT.__init__)
    valid_params = sig.parameters.keys()
    filtered_kwargs = {
        k: v for k, v in raw_kwargs.items() if k in valid_params
    }

    print("[build_model] VideoViT init kwargs:")
    for k, v in filtered_kwargs.items():
        print(f"    {k} = {v}")

    model = VideoViT(**filtered_kwargs)
    model.to(device)
    return model


# ------------------------------
#  构建 DataLoader
# ------------------------------

def build_dataloaders(cfg):
    data_cfg = cfg.data
    root = Path(data_cfg.root)
    frame_root = root / "frames"

    train_split = Path(data_cfg.train_split)
    val_split = Path(data_cfg.val_split)

    clip_len = int(data_cfg.clip_len)
    img_size = int(data_cfg.img_size)

    # 先扫描 train + val 的类名，构建统一 label_to_idx
    cls_names = set()
    for split_file in [train_split, val_split]:
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.split()[0]
                cls_name = rel_path.split("/")[0]
                cls_names.add(cls_name)

    cls_names = sorted(cls_names)
    label_to_idx = {c: i for i, c in enumerate(cls_names)}
    print(f"[dataset] num_classes from splits = {len(label_to_idx)}")

    train_ds = UCF101FramesDataset(
        frame_root=frame_root,
        split_file=train_split,
        label_to_idx=label_to_idx,
        clip_len=clip_len,
        img_size=img_size,
        is_train=True,
    )
    val_ds = UCF101FramesDataset(
        frame_root=frame_root,
        split_file=val_split,
        label_to_idx=label_to_idx,
        clip_len=clip_len,
        img_size=img_size,
        is_train=False,
    )

    batch_size = int(cfg.train.batch_size)
    num_workers = int(data_cfg.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ------------------------------
#  训练 & 验证
# ------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for video, label in dataloader:
            # video: [B, T, C, H, W] ，我们按此约定
            video = video.to(device, non_blocking=True)
            label = torch.as_tensor(label, device=device)

            # 如果 VideoViT 期望 [B, C, T, H, W]，可以在这里 permute
            logits = model(video)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.numel()

    acc = correct / max(total, 1)
    return acc


def train_one_epoch(
    model,
    proxy: LayerHwProxy,
    train_loader,
    optimizer,
    scaler: GradScaler,
    cfg,
    device,
    epoch: int,
):
    model.train()
    lambda_sparse = float(cfg.loss.lambda_sparse)
    lambda_hw = float(cfg.loss.lambda_hw)
    alpha_t = float(cfg.loss.alpha_t)
    alpha_m = float(cfg.loss.alpha_m)
    alpha_e = float(cfg.loss.alpha_e)

    use_hw_loss = bool(cfg.proxy.use_hw_loss)
    device_name = cfg.proxy.device_name

    # 简单的归一化常数，防止 hw_loss 量级太大
    LAT_NORM = 10.0
    MEM_NORM = 2000.0
    ENE_NORM = 10.0

    print_freq = int(cfg.log.print_freq)

    for it, (video, label) in enumerate(train_loader):
        video = video.to(device, non_blocking=True)
        label = torch.as_tensor(label, device=device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=bool(cfg.train.amp)):
            logits = model(video)  # [B, num_classes]
            cls_loss = F.cross_entropy(logits, label)

            # 1) 稀疏 / 时空熵正则
            sparse_loss = entropy_utils.compute_entropy_loss(model, video)

            # 2) 硬件 loss
            if use_hw_loss:
                if hasattr(model, "get_keep_ratios") and hasattr(
                    model, "get_layer_metas"
                ):
                    keep_ratios = model.get_keep_ratios()
                    layer_metas = model.get_layer_metas(video.shape, keep_ratios)
                else:
                    layer_metas = []

                hw_cost = 0.0
                for meta in layer_metas:
                    pred = proxy.predict_layer(device_name, meta)
                    hw_cost = hw_cost + (
                        alpha_t * (pred["ms"] / LAT_NORM)
                        + alpha_m * (pred["mem"] / MEM_NORM)
                        + alpha_e * (pred["energy"] / ENE_NORM)
                    )
                hw_loss = hw_cost
            else:
                sparse_loss = sparse_loss * 0.0
                hw_loss = torch.zeros(1, device=device)

            loss = cls_loss + lambda_sparse * sparse_loss + lambda_hw * hw_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (it + 1) % print_freq == 0:
            print(
                f"[Epoch {epoch}] Iter [{it+1}/{len(train_loader)}] "
                f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                f"sparse={float(sparse_loss):.4f} hw={float(hw_loss):.4f}"
            )


# ------------------------------
#  主函数
# ------------------------------

def main():
    # 默认配置路径
    cfg_path = "./configs/ast2_ucf101.yaml"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        cfg_path = sys.argv[1]

    cfg = load_config(cfg_path)
    print(f"[config] use: {cfg_path}")

    # 设备
    device_str = cfg.train.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[device] training device: {device}")

    # 随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 日志目录
    out_dir = Path(cfg.log.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) DataLoader
    train_loader, val_loader = build_dataloaders(cfg)

    # 2) 模型
    model = build_model(cfg, device)

    # 3) 硬件代理（这里用正确的参数名 gpu_yaml / proxy_weight_dir）
    proxy = LayerHwProxy(
        gpu_yaml=cfg.proxy.gpu_yaml,
        proxy_weight_dir=cfg.proxy.weight_dir,
    )
    print(
        f"[LayerHwProxy] gpu_yaml={cfg.proxy.gpu_yaml}, "
        f"weight_dir={cfg.proxy.weight_dir}"
    )

    # 4) 优化器 & 调度器
    lr = float(cfg.train.lr)
    weight_decay = float(getattr(cfg.train, "weight_decay", 0.0))
    print(f"[optimizer] lr={lr}, weight_decay={weight_decay}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    epochs = int(cfg.train.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    scaler = GradScaler(enabled=bool(cfg.train.amp))

    best_acc = 0.0
    ckpt_path = Path(cfg.log.ckpt_path)

    for epoch in range(epochs):
        print(f"\n==== Epoch {epoch}/{epochs-1} ====")
        train_one_epoch(
            model, proxy, train_loader, optimizer, scaler, cfg, device, epoch
        )

        scheduler.step()

        # 验证
        val_acc = evaluate(model, val_loader, device)
        print(f"[val] epoch={epoch} acc={val_acc:.4f}")

        # 保存 best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg_path": cfg_path,
                },
                ckpt_path,
            )
            print(f"[ckpt] new best_acc={best_acc:.4f}, saved to {ckpt_path}")

    print(f"\n[done] training finished. best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
