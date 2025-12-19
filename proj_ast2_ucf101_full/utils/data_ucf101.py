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
    t_start: int
    cover_len: int
    is_video: bool
    video_id: str
    total_frames: int


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

        self.split = split
        clip_lens = getattr(cfg.data, "clip_lens", None)
        clip_len = getattr(cfg.data, "clip_len", None)
        num_frames = getattr(cfg.data, "num_frames", None)
        if clip_lens is None and clip_len is not None:
            clip_lens = [int(clip_len)]
            if num_frames is None:
                num_frames = int(clip_len)
        if clip_lens is None:
            raise ValueError("Expected data.clip_lens or data.clip_len to be set.")
        if num_frames is None:
            raise ValueError("Expected data.num_frames to be set.")
        self.clip_lens = [int(c) for c in clip_lens]
        self.num_frames = int(num_frames)
        self.use_audio = bool(getattr(cfg.data, "use_audio", False))
        audio_root = getattr(cfg.data, "audio_root", None)
        self.audio_root = Path(audio_root) if audio_root else (self.root / "audio")
        self.audio_feat_dim = int(getattr(cfg.data, "audio_feat_dim", cfg.audio.feat_dim))
        self._audio_missing_warned = False
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
        self._frame_cache: Dict[str, List[Path]] = {}
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
                    video_id = source_path.stem
                else:
                    video_stem = Path(rel_path).stem
                    source_path = self.root / cls_name / video_stem
                    if not source_path.is_dir():
                        continue
                    frame_files = sorted(p for p in source_path.glob("*.jpg") if p.is_file())
                    if not frame_files:
                        continue
                    total_frames = len(frame_files)
                    video_id = video_stem
                    self._frame_cache[video_id] = frame_files
                stride_ratio = cfg.data.train_stride_ratio if self.is_train else cfg.data.eval_stride_ratio
                for cover_len in self.clip_lens:
                    stride = max(1, int(cover_len * stride_ratio))
                    offset = 0
                    if self.is_train and getattr(cfg.data, "clip_jitter", False):
                        offset = random.randint(0, max(0, stride - 1))
                    for start in range(offset, total_frames, stride):
                        t_start = start
                        if t_start >= total_frames:
                            break
                        clips.append(
                            ClipItem(
                                source_path=source_path,
                                label=label,
                                t_start=t_start,
                                cover_len=cover_len,
                                is_video=self.mode == "video",
                                video_id=video_id,
                                total_frames=total_frames,
                            )
                        )
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

    def _load_frames(self, source_path: Path, frame_indices: List[int], is_video: bool, video_id: str) -> List[Image.Image]:
        if is_video:
            frames, _, _ = read_video(str(source_path), pts_unit="sec")
            if frames.numel() == 0:
                img = Image.new("RGB", (self.img_size, self.img_size))
                return [img for _ in range(len(frame_indices))]
            total_frames = frames.shape[0]
            images: List[Image.Image] = []
            for i in frame_indices:
                i = min(i, total_frames - 1)
                frame = frames[i].cpu().numpy()
                images.append(Image.fromarray(frame))
            return images
        frame_files = self._frame_cache.get(video_id)
        if frame_files is None:
            frame_files = sorted(p for p in source_path.glob("*.jpg") if p.is_file())
        if not frame_files:
            img = Image.new("RGB", (self.img_size, self.img_size))
            return [img for _ in range(len(frame_indices))]
        images: List[Image.Image] = []
        for i in frame_indices:
            i = min(i, len(frame_files) - 1)
            with Image.open(frame_files[i]) as img:
                images.append(img.convert("RGB"))
        return images

    def _load_audio_clip(self, video_id: str, frame_indices: List[int]) -> Optional[torch.Tensor]:
        if not self.use_audio:
            return None
        audio_path = self.audio_root / f"{video_id}.npy"
        if not audio_path.is_file():
            if not self._audio_missing_warned:
                print(f"[WARN] audio feature missing at {audio_path}, returning zeros.")
                self._audio_missing_warned = True
            return torch.zeros(len(frame_indices), self.audio_feat_dim, dtype=torch.float32)
        data = np.load(audio_path)
        if data.ndim != 2:
            raise ValueError(f"Audio feature must be 2D [T, D], got {data.shape} for {audio_path}")
        max_index = max(frame_indices) if frame_indices else 0
        if data.shape[0] <= max_index:
            pad_len = max_index + 1 - data.shape[0]
            pad = np.repeat(data[-1:, :], pad_len, axis=0)
            data = np.concatenate([data, pad], axis=0)
        clip = data[frame_indices]
        return torch.from_numpy(clip).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.clips[idx]
        t_end = min(item.t_start + item.cover_len, item.total_frames)
        window_indices = list(range(item.t_start, t_end))
        if len(window_indices) >= self.num_frames:
            select = np.linspace(0, len(window_indices) - 1, self.num_frames)
            select = np.round(select).astype(int).tolist()
            frame_indices = [window_indices[i] for i in select]
        else:
            frame_indices = window_indices.copy()
            if not frame_indices:
                frame_indices = [max(0, item.total_frames - 1)]
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
        frames = self._load_frames(item.source_path, frame_indices, item.is_video, item.video_id)
        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sample = {
            "video": video,
            "label": torch.tensor(item.label, dtype=torch.long),
            "video_id": item.video_id,
        }
        audio = self._load_audio_clip(item.video_id, frame_indices)
        if audio is not None:
            sample["audio"] = audio
        return sample
