"""UCF101 dataset with multi-scale sliding windows and optional audio."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class ClipItem:
    frames_dir: Path
    label: int
    start: int
    clip_len: int


def sample_frame_indices(num_total: int, num_frames: int, mode: str = "uniform") -> np.ndarray:
    """Sample fixed-length frame indices for a clip."""
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
        indices = list(range(num_total))
        while len(indices) < num_frames:
            indices.append(num_total - 1)
        indices = np.array(indices, dtype=np.int64)
    return indices


class UCF101Dataset(Dataset):
    def __init__(self, cfg, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.frame_root = self.root / "frames"
        if not self.frame_root.exists():
            self.frame_root = self.root
        if self.frame_root == self.root:
            self.audio_root = self.root.parent / "audio"
        else:
            self.audio_root = self.root / "audio"
        self.split = split
        self.modality = cfg.data.modality
        self.is_train = split == "train"
        self.img_size = cfg.data.img_size
        self.num_frames = getattr(cfg.data, "num_frames", cfg.data.clip_len)
        self.frame_sampling = getattr(cfg.data, "frame_sampling", "uniform")

        split_file = Path(cfg.data.train_split if self.is_train else cfg.data.val_split)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        label_to_idx: Dict[str, int] = {}
        clips: List[ClipItem] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.split()[0]
                cls_name = rel_path.split("/")[0]
                if cls_name not in label_to_idx:
                    label_to_idx[cls_name] = len(label_to_idx)
                label = label_to_idx[cls_name]
                video_stem = Path(rel_path).stem
                frames_dir = self.frame_root / cls_name / video_stem
                if frames_dir.is_dir():
                    clips.append(ClipItem(frames_dir, label, 0, self.num_frames))
        self.clips = clips

        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(self.img_size * 1.14)),
                    transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(self.img_size * 1.14)),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frames(self, frames_dir: Path, clip_len: int) -> List[Image.Image]:
        frame_files = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
        if not frame_files:
            img = Image.new("RGB", (self.img_size, self.img_size))
            return [img for _ in range(clip_len)]
        mode = self.frame_sampling if self.is_train else "center"
        indices = sample_frame_indices(len(frame_files), clip_len, mode=mode)
        images: List[Image.Image] = []
        for i in indices:
            with Image.open(frame_files[int(i)]) as img:
                images.append(img.convert("RGB"))
        return images

    def _load_audio(self, frames_dir: Path, clip_len: int) -> Optional[torch.Tensor]:
        if self.modality != "video_audio":
            return None
        audio_path = self.audio_root / f"{frames_dir.name}.npy"
        if not audio_path.is_file():
            return torch.zeros(clip_len, self.cfg.audio.feat_dim, dtype=torch.float32)
        data = np.load(audio_path)
        if data.ndim != 2:
            raise ValueError(f"Audio feature must be 2D [T, D], got {data.shape} for {audio_path}")
        if data.shape[0] < clip_len:
            pad_len = clip_len - data.shape[0]
            pad = np.repeat(data[-1:, :], pad_len, axis=0)
            data = np.concatenate([data, pad], axis=0)
        clip = data[:clip_len]
        return torch.from_numpy(clip).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.clips[idx]
        frames = self._load_frames(item.frames_dir, self.num_frames)
        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        if video.shape[0] != self.num_frames:
            if video.shape[0] > self.num_frames:
                video = video[: self.num_frames]
            else:
                pad = video[-1:].repeat(self.num_frames - video.shape[0], 1, 1, 1)
                video = torch.cat([video, pad], dim=0)
        sample = {"video": video, "label": torch.tensor(item.label, dtype=torch.long)}
        audio = self._load_audio(item.frames_dir, self.num_frames)
        if audio is not None:
            sample["audio"] = audio
        return sample
