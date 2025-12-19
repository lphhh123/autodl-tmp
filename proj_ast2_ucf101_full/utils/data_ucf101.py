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
from torchvision.io import read_video, read_video_timestamps


@dataclass
class ClipItem:
    source_path: Path
    label: int
    start: int
    clip_len: int
    is_video: bool


class UCF101Dataset(Dataset):
    def __init__(self, cfg, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg.data
        frames_root = getattr(data_cfg, "frames_root", None)
        video_root = getattr(data_cfg, "video_root", None)
        root = getattr(data_cfg, "root", None)
        self.mode = "frames"
        if frames_root:
            self.root = Path(frames_root)
        elif video_root:
            self.root = Path(video_root)
            self.mode = "video"
        elif root:
            self.root = Path(root)
        else:
            raise ValueError("Expected data.frames_root, data.video_root, or data.root to be set.")

        self.audio_root = self.root / "audio"
        self.split = split
        self.clip_lens = list(cfg.data.clip_lens)
        self.modality = cfg.data.modality
        self.is_train = split == "train"
        self.img_size = cfg.data.img_size
        self.splits_root = Path(getattr(data_cfg, "splits_root", ""))
        split_file = Path(
            cfg.data.train_split if self.is_train else cfg.data.val_split
        )
        if not split_file.is_absolute() and self.splits_root:
            split_file = self.splits_root / split_file
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file} (splits_root={self.splits_root})"
            )

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
                video_name = Path(rel_path).name
                if self.mode == "video":
                    source_path = self.root / cls_name / video_name
                    if not source_path.is_file():
                        continue
                    total_frames = self._get_video_num_frames(source_path)
                    if total_frames <= 0:
                        continue
                else:
                    video_stem = Path(rel_path).stem
                    source_path = self.root / cls_name / video_stem
                    if not source_path.is_dir():
                        continue
                    frame_files = sorted(p for p in source_path.glob("*.jpg") if p.is_file())
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
                            clips.append(ClipItem(source_path, label, start, clip_len, self.mode == "video"))
                        else:
                            clips.append(
                                ClipItem(
                                    source_path,
                                    label,
                                    max(0, total_frames - clip_len),
                                    clip_len,
                                    self.mode == "video",
                                )
                            )
                            break
        self.clips = clips
        if len(self.clips) == 0:
            raise RuntimeError(
                "UCF101Dataset has 0 samples. "
                f"mode={self.mode}, root={self.root}, splits_root={self.splits_root}, split={split_file}"
            )

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

    def _get_video_num_frames(self, video_path: Path) -> int:
        try:
            pts, _ = read_video_timestamps(str(video_path), pts_unit="sec")
            return len(pts)
        except Exception:
            frames, _, _ = read_video(str(video_path), pts_unit="sec")
            return int(frames.shape[0])

    def _load_frames(self, source_path: Path, start: int, clip_len: int, is_video: bool) -> List[Image.Image]:
        if is_video:
            frames, _, _ = read_video(str(source_path), pts_unit="sec")
            if frames.numel() == 0:
                img = Image.new("RGB", (self.img_size, self.img_size))
                return [img for _ in range(clip_len)]
            total_frames = frames.shape[0]
            indices = list(range(start, start + clip_len))
            indices = [min(i, total_frames - 1) for i in indices]
            images: List[Image.Image] = []
            for i in indices:
                frame = frames[i].cpu().numpy()
                images.append(Image.fromarray(frame))
            return images
        frame_files = sorted(p for p in source_path.glob("*.jpg") if p.is_file())
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

    def _load_audio(self, source_path: Path, start: int, clip_len: int) -> Optional[torch.Tensor]:
        if self.modality != "video_audio":
            return None
        audio_path = self.audio_root / f"{source_path.stem}.npy"
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
        frames = self._load_frames(item.source_path, item.start, item.clip_len, item.is_video)
        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sample = {"video": video, "label": torch.tensor(item.label, dtype=torch.long)}
        audio = self._load_audio(item.source_path, item.start, item.clip_len)
        if audio is not None:
            sample["audio"] = audio
        return sample
