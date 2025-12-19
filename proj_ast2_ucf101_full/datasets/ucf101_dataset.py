"""UCF101 frame-based dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class UCF101Sample:
    frames_dir: Path
    label: int


def sample_frame_indices(num_total: int, num_frames: int, mode: str = "uniform") -> np.ndarray:
    if num_total <= 0:
        raise ValueError(f"num_total must be positive, got {num_total}")
    if num_total >= num_frames:
        if mode == "uniform":
            indices = np.linspace(0, num_total - 1, num_frames).astype(np.int64)
        elif mode == "random":
            start = random.randint(0, num_total - num_frames)
            indices = np.arange(start, start + num_frames, dtype=np.int64)
        elif mode == "center":
            center = (num_total - 1) / 2.0
            start = int(round(center - (num_frames - 1) / 2.0))
            start = max(0, min(start, num_total - num_frames))
            indices = np.arange(start, start + num_frames, dtype=np.int64)
        else:
            raise ValueError(f"Unknown frame sampling mode: {mode}")
    else:
        base = list(range(num_total))
        while len(base) < num_frames:
            base.append(num_total - 1)
        indices = np.array(base, dtype=np.int64)
    assert len(indices) == num_frames
    return indices


class UCF101Dataset(Dataset):
    """Frame-based UCF101 dataset.

    Parameters
    ----------
    root: str or Path
        Dataset root containing ``frames`` and ``splits``.
    split_file: str or Path
        Split text such as ``trainlist01.txt`` or ``testlist01.txt``.
    clip_len: int
        Number of frames per clip.
    img_size: int
        Square image size after resize/crop.
    is_train: bool
        Whether to apply training augmentations.
    transform: Callable
        Optional transform applied to each frame. If ``None`` a default
        pipeline is used.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        split_file: Path | str | None = None,
        clip_len: int = 8,
        img_size: int = 224,
        is_train: bool = True,
        transform: Callable | None = None,
        cfg: Dict | None = None,
        split: str | None = None,
    ) -> None:
        super().__init__()
        if cfg is not None:
            data_cfg = cfg["data"]
            split = split or ("train" if is_train else "val")
            root = data_cfg.get("frames_root") or data_cfg.get("video_root") or data_cfg.get("root")
            splits_root = Path(data_cfg.get("splits_root", ""))
            if split == "train":
                split_file = data_cfg.get("split_file_train", data_cfg.get("train_split"))
            else:
                split_file = data_cfg.get("split_file_test", data_cfg.get("val_split"))
            if split_file and splits_root and not Path(split_file).is_absolute():
                split_file = splits_root / split_file
            self.clip_len = int(data_cfg.get("num_frames", clip_len))
            self.is_train = split == "train"
            self.frame_sampling = data_cfg.get("frame_sampling", "uniform")
        else:
            self.clip_len = clip_len
            self.is_train = is_train
            self.frame_sampling = "uniform"

        if root is None or split_file is None:
            raise ValueError("root and split_file must be provided when cfg is None.")

        self.root = Path(root)
        self.frame_root = self.root
        self.img_size = img_size

        self.samples: List[UCF101Sample] = []
        split_path = Path(split_file)
        label_to_idx: Dict[str, int] = {}

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                cls_name = rel_path.split("/")[0]
                if cls_name not in label_to_idx:
                    label_to_idx[cls_name] = len(label_to_idx)
                label = label_to_idx[cls_name]
                video_stem = Path(rel_path).stem
                frames_dir = self.frame_root / cls_name / video_stem
                if frames_dir.is_dir():
                    self.samples.append(UCF101Sample(frames_dir=frames_dir, label=label))

        if transform is not None:
            self.transform = transform
        else:
            if is_train:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(int(img_size * 1.14)),
                        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, frames_dir: Path) -> List[Image.Image]:
        frame_files = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
        if len(frame_files) == 0:
            return []
        mode = self.frame_sampling if self.is_train else "center"
        indices = sample_frame_indices(len(frame_files), self.clip_len, mode=mode)
        selected = [frame_files[int(i)] for i in indices]

        images: List[Image.Image] = []
        for file in selected:
            with Image.open(file) as img:
                images.append(img.convert("RGB"))
        return images

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        frames = self._load_frames(sample.frames_dir)
        if len(frames) == 0:
            img = Image.new("RGB", (self.img_size, self.img_size))
            frames = [img for _ in range(self.clip_len)]

        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0).contiguous().clone()  # [T, C, H, W]
        label = torch.tensor(sample.label, dtype=torch.long)
        return {"video": video, "label": label}


def collate_video_batch(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function returning ``[B, T, C, H, W]`` videos and labels."""
    videos, labels = zip(*batch)
    video_tensor = torch.stack(videos, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return video_tensor, label_tensor
