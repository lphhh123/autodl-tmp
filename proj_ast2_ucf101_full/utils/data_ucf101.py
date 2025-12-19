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


class UCF101Dataset(Dataset):
    def __init__(self, cfg, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.frame_root = self.root / "frames"
        self.audio_root = self.root / "audio"
        self.split = split
        self.clip_lens = list(cfg.data.clip_lens)
        self.modality = cfg.data.modality
        self.is_train = split == "train"
        self.img_size = cfg.data.img_size

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
                if not frames_dir.is_dir():
                    continue
                frame_files = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
                if not frame_files:
                    continue
                total_frames = len(frame_files)
                for clip_len in self.clip_lens:
                    stride_ratio = cfg.data.train_stride_ratio if self.is_train else cfg.data.eval_stride_ratio
                    stride = max(1, int(clip_len * stride_ratio))
                    offset = 0
                    if self.is_train and getattr(cfg.data, "clip_jitter", False):
                        offset = random.randint(0, max(0, stride - 1))
                    for start in range(offset, total_frames, stride):
                        if start + clip_len <= total_frames:
                            clips.append(ClipItem(frames_dir, label, start, clip_len))
                        else:
                            clips.append(ClipItem(frames_dir, label, max(0, total_frames - clip_len), clip_len))
                            break
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

    def _load_frames(self, frames_dir: Path, start: int, clip_len: int) -> List[Image.Image]:
        frame_files = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
        if not frame_files:
            img = Image.new("RGB", (self.img_size, self.img_size))
            return [img for _ in range(clip_len)]
        indices = list(range(start, start + clip_len))
        indices = [min(i, len(frame_files) - 1) for i in indices]
        images: List[Image.Image] = []
        for i in indices:
            with Image.open(frame_files[i]) as img:
                images.append(img.convert("RGB"))
        return images

    def _load_audio(self, frames_dir: Path, start: int, clip_len: int) -> Optional[torch.Tensor]:
        if self.modality != "video_audio":
            return None
        audio_path = self.audio_root / f"{frames_dir.name}.npy"
        if not audio_path.is_file():
            return torch.zeros(clip_len, self.cfg.audio.feat_dim, dtype=torch.float32)
        data = np.load(audio_path)
        if data.ndim != 2:
            raise ValueError(f"Audio feature must be 2D [T, D], got {data.shape} for {audio_path}")
        if data.shape[0] < start + clip_len:
            pad_len = start + clip_len - data.shape[0]
            pad = np.repeat(data[-1:, :], pad_len, axis=0)
            data = np.concatenate([data, pad], axis=0)
        clip = data[start : start + clip_len]
        return torch.from_numpy(clip).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.clips[idx]
        frames = self._load_frames(item.frames_dir, item.start, item.clip_len)
        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sample = {"video": video, "label": torch.tensor(item.label, dtype=torch.long)}
        audio = self._load_audio(item.frames_dir, item.start, item.clip_len)
        if audio is not None:
            sample["audio"] = audio
        return sample
