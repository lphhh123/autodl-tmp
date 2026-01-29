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


def _entropy_gray_frame(frame_path: Path, downsample: int = 32, bins: int = 32) -> float:
    try:
        with Image.open(frame_path) as img:
            gray = img.convert("L")
            if downsample and downsample > 0:
                gray = gray.resize((downsample, downsample))
            arr = np.asarray(gray, dtype=np.uint8)
    except (OSError, ValueError):
        return 1.0

    if arr.size == 0:
        return 1.0
    hist, _ = np.histogram(arr.flatten(), bins=bins, range=(0, 255))
    total = hist.sum()
    if total <= 0:
        return 1.0
    prob = hist.astype(np.float64) / float(total)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    if not np.isfinite(entropy) or entropy <= 0:
        return 1.0
    return float(entropy)


def _entropy_gray_diff(
    frame_path_a: Path, frame_path_b: Path, downsample: int = 32, bins: int = 32
) -> float:
    try:
        with Image.open(frame_path_a) as ia:
            a = ia.convert("L")
            if downsample and downsample > 0:
                a = a.resize((downsample, downsample))
            a = np.asarray(a, dtype=np.int16)
        with Image.open(frame_path_b) as ib:
            b = ib.convert("L")
            if downsample and downsample > 0:
                b = b.resize((downsample, downsample))
            b = np.asarray(b, dtype=np.int16)
    except (OSError, ValueError):
        return 1.0
    if a.size == 0 or b.size == 0:
        return 1.0
    diff = np.abs(a - b).astype(np.uint8)
    hist, _ = np.histogram(diff.flatten(), bins=bins, range=(0, 255))
    total = hist.sum()
    if total <= 0:
        return 1.0
    prob = hist.astype(np.float64) / float(total)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    if not np.isfinite(entropy) or entropy <= 0:
        return 1.0
    return float(entropy)


def _retain_starts_entropy_density_train(
    starts: List[int],
    cover_len: int,
    frame_files: List[Path],
    total_frames: int,
    video_id: str,
    cfg,
    hglobal_cache: Optional[Dict[str, float]],
) -> List[int]:
    if not starts:
        return starts
    ret = getattr(cfg.data, "window_retention", None)
    if ret is None or not bool(getattr(ret, "enabled", False)):
        return starts

    max_windows = int(getattr(ret, "max_windows_per_video_per_len", 4))
    if max_windows <= 0:
        return []

    eval_candidates = int(getattr(ret, "eval_candidates", 128))
    global_frames = int(getattr(ret, "global_frames", 8))
    lambda_dist = float(getattr(ret, "lambda_dist", 0.5))
    temporal_delta = int(getattr(ret, "temporal_delta", 1))
    downsample = int(getattr(ret, "downsample", 32))
    bins = int(getattr(ret, "entropy_bins", 32))
    space_weight = float(getattr(ret, "space_weight", getattr(ret, "alpha", 0.5)))
    time_weight = float(getattr(ret, "time_weight", 1.0 - space_weight))
    min_gap = int(getattr(ret, "min_gap", max(1, cover_len // 2)))

    T = int(total_frames)
    if T <= 0:
        return starts

    cache_key = str(video_id)
    if (hglobal_cache is not None) and (cache_key in hglobal_cache):
        H_global = float(hglobal_cache[cache_key])
    else:
        gk = max(1, min(global_frames, T))
        g_idxs = np.linspace(0, T - 1, gk)
        g_idxs = np.floor(g_idxs).astype(int).tolist()
        g_es = []
        for fi in g_idxs:
            fi = max(0, min(fi, T - 1))
            t_idx = max(0, min(fi + temporal_delta, T - 1))
            Hs = _entropy_gray_frame(frame_files[fi], downsample=downsample, bins=bins)
            Ht = _entropy_gray_diff(
                frame_files[fi], frame_files[t_idx], downsample=downsample, bins=bins
            )
            g_es.append(space_weight * Hs + time_weight * Ht)
        H_global = float(np.mean(g_es)) if g_es else 1.0
        if (not np.isfinite(H_global)) or H_global <= 0:
            H_global = 1.0
        if hglobal_cache is not None:
            hglobal_cache[cache_key] = float(H_global)

    if len(starts) > eval_candidates:
        idxs = np.linspace(0, len(starts) - 1, eval_candidates)
        idxs = np.floor(idxs).astype(int).tolist()
        candidates = [starts[i] for i in idxs]
    else:
        candidates = list(starts)

    center_offsets = [min(max(int(s + cover_len // 2), 0), T - 1) for s in candidates]
    H_locals: List[float] = []
    for fi in center_offsets:
        t_idx = max(0, min(fi + temporal_delta, T - 1))
        Hs = _entropy_gray_frame(frame_files[fi], downsample=downsample, bins=bins)
        Ht = _entropy_gray_diff(frame_files[fi], frame_files[t_idx], downsample=downsample, bins=bins)
        H_locals.append(space_weight * Hs + time_weight * Ht)

    if not H_locals:
        return []

    anchor_idx = int(np.argmax(H_locals))
    anchor_start = candidates[anchor_idx]

    rho_scores: List[float] = []
    for start, H_local in zip(candidates, H_locals):
        dist_norm = abs(start - anchor_start) / max(cover_len, 1)
        ratio = H_local / max(H_global, 1e-6)
        rho_scores.append(ratio * math.exp(-lambda_dist * dist_norm))

    order = sorted(range(len(candidates)), key=lambda i: rho_scores[i], reverse=True)
    selected: List[int] = []
    for idx in order:
        start = candidates[idx]
        if not selected:
            selected.append(start)
            if len(selected) >= max_windows:
                break
            continue
        if min(abs(start - s) for s in selected) >= min_gap:
            selected.append(start)
            if len(selected) >= max_windows:
                break

    if len(selected) < max_windows:
        for idx in order:
            start = candidates[idx]
            if start in selected:
                continue
            selected.append(start)
            if len(selected) >= max_windows:
                break

    return sorted(selected)


def _try_load_classind(root: Path) -> Optional[Dict[str, int]]:
    """
    Try to load UCF101 classInd.txt (1-based indices).
    Returns mapping {class_name: idx0_based}.
    """
    cand = [
        root / "classInd.txt",
        root / "splits" / "classInd.txt",
        root / "ucfTrainTestlist" / "classInd.txt",
    ]
    for p in cand:
        if p.exists():
            mapping: Dict[str, int] = {}
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    cls = parts[1]
                    mapping[cls] = int(parts[0]) - 1
            if mapping:
                return mapping
    return None


def _parse_class_names_from_list(list_path: Path) -> List[str]:
    names: List[str] = []
    if not list_path.exists():
        return names
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0]
            cls = rel.split("/")[0]
            names.append(cls)
    return names


def _build_deterministic_label_map(root: Path, split_list: Path) -> Dict[str, int]:
    """
    Deterministic mapping for both train/test.
    Priority:
      1) classInd.txt
      2) union(trainlist*, testlist*) in same folder as split_list (if exists)
      3) union only this split file (fallback) but sorted
    """
    classind = _try_load_classind(root)
    if classind:
        return classind

    folder = split_list.parent
    train_lists = sorted(folder.glob("trainlist*.txt"))
    test_lists = sorted(folder.glob("testlist*.txt"))
    all_names: List[str] = []
    if train_lists and test_lists:
        for p in train_lists + test_lists:
            all_names.extend(_parse_class_names_from_list(p))
    else:
        all_names.extend(_parse_class_names_from_list(split_list))

    uniq = sorted(set(all_names))
    return {c: i for i, c in enumerate(uniq)}


@dataclass
class ClipItem:
    source_path: Path
    label: int
    t_start: int
    cover_len: int
    video_id: str
    total_frames: int


class UCF101Dataset(Dataset):
    def __init__(self, cfg, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg.data
        repo_root = Path(__file__).resolve().parents[1]
        workspace_root = repo_root.parent
        frames_root = Path(getattr(data_cfg, "frames_root", "data/ucf101/frames"))
        splits_root = Path(getattr(data_cfg, "splits_root", "data/ucf101/splits"))
        audio_root = getattr(data_cfg, "audio_root", None)
        if audio_root is not None:
            audio_root = Path(audio_root)

        def resolve_root(path_value: Path) -> Path:
            if path_value.is_absolute():
                return path_value
            repo_candidate = repo_root / path_value
            if repo_candidate.exists():
                return repo_candidate
            workspace_candidate = workspace_root / path_value
            if workspace_candidate.exists():
                return workspace_candidate
            return repo_candidate

        frames_root = resolve_root(frames_root)
        splits_root = resolve_root(splits_root)
        if audio_root is not None:
            audio_root = resolve_root(audio_root)

        self.root = frames_root
        self.splits_root = splits_root

        requested_split = str(split).lower()
        if requested_split in ("val", "valid", "validation"):
            split_label = "val"
            split_mode = "test"
        else:
            split_label = requested_split
            split_mode = requested_split
        self.split = split_mode
        self.is_train = self.split == "train"
        if self.is_train and getattr(cfg.data, "clip_lens_train", None) is not None:
            clip_lens = getattr(cfg.data, "clip_lens_train")
        elif (not self.is_train) and getattr(cfg.data, "clip_lens_eval", None) is not None:
            clip_lens = getattr(cfg.data, "clip_lens_eval")
        else:
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
        self.audio_root = Path(audio_root) if audio_root is not None else (self.root / "audio")
        self.audio_feat_dim = int(getattr(cfg.data, "audio_feat_dim", cfg.audio.feat_dim))
        self._audio_missing_warned = False
        self.img_size = cfg.data.img_size
        if split_label == "train":
            split_name = getattr(cfg.data, "train_split", "trainlist01.txt")
        elif split_label == "val":
            split_name = getattr(cfg.data, "val_split", getattr(cfg.data, "test_split", "testlist01.txt"))
        else:
            split_name = getattr(cfg.data, "test_split", getattr(cfg.data, "val_split", "testlist01.txt"))
        split_path = Path(split_name)
        tried_split_files: List[Path] = []
        if split_path.is_absolute():
            split_file = split_path
        elif len(split_path.parts) == 1:
            split_file = self.splits_root / split_path
        else:
            split_file = repo_root / split_path
        tried_split_files.append(split_file)
        self.split_file = split_file
        if not split_file.is_file():
            existing_txt = sorted(self.splits_root.glob("*.txt"))
            existing_preview = [str(p) for p in existing_txt[:30]]
            tried_paths = [str(p.resolve()) for p in tried_split_files]
            raise FileNotFoundError(
                "Split file not found for split="
                f"{split_label}. repo_root={repo_root}, workspace_root={workspace_root}, "
                f"effective_splits_root={self.splits_root}, tried_split_files={tried_paths}, "
                f"splits_root_txt_files={existing_preview}. "
                "Ensure data exists under repo_root/data/ucf101 or workspace_root/data/ucf101, "
                "or update cfg.data.splits_root/train_split/val_split/test_split."
            )
        print(
            "[UCF101Dataset DEBUG] "
            f"repo_root={repo_root}, workspace_root={workspace_root}, frames_root={self.root}, "
            f"splits_root={self.splits_root}, split_file={split_file}, audio_root={self.audio_root}, "
            f"exists(frames)={self.root.exists()}, exists(splits)={self.splits_root.exists()}, "
            f"exists(split_file)={split_file.exists()}"
        )

        self.label_to_idx = _build_deterministic_label_map(self.root, split_file)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        clips: List[ClipItem] = []
        self._frame_cache: Dict[str, List[Path]] = {}
        self._hglobal_cache: Dict[str, float] = {}
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.split()[0]
                cls_name = rel_path.split("/")[0]
                if cls_name not in self.label_to_idx:
                    continue
                label = self.label_to_idx[cls_name]
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
                    if total_frames <= cover_len:
                        starts = [0]
                    else:
                        offset = 0
                        if self.is_train and getattr(cfg.data, "clip_jitter", False):
                            offset = random.randint(0, max(0, stride - 1))
                        max_start = max(total_frames - cover_len + 1, 1)
                        starts = list(range(offset, max_start, stride))
                    if self.is_train:
                        starts = _retain_starts_entropy_density_train(
                            starts=starts,
                            cover_len=int(cover_len),
                            frame_files=frame_files,
                            total_frames=int(total_frames),
                            video_id=video_id,
                            cfg=cfg,
                            hglobal_cache=self._hglobal_cache,
                        )
                    for start in starts:
                        t_start = start
                        clips.append(
                            ClipItem(
                                source_path=source_path,
                                label=label,
                                t_start=t_start,
                                cover_len=cover_len,
                                video_id=video_id,
                                total_frames=total_frames,
                            )
                        )
        self.clips = clips
        if len(self.clips) == 0:
            raise RuntimeError(
                "UCF101Dataset has 0 samples. "
                f"root={self.root}, splits_root={self.splits_root}, split={split_file}"
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

    def _load_frames(self, source_path: Path, frame_indices: List[int], video_id: str) -> List[Image.Image]:
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
        total_frames = item.total_frames
        t_end = min(item.t_start + item.cover_len, total_frames)
        t_end = max(t_end, item.t_start)
        Lw = max(0, t_end - item.t_start)
        if Lw >= self.num_frames:
            idxs = np.linspace(0, Lw - 1, self.num_frames)
            idxs = np.floor(idxs).astype(int)
        else:
            base = np.arange(Lw, dtype=int)
            if base.size == 0:
                idxs = np.zeros(self.num_frames, dtype=int)
            else:
                pad_needed = self.num_frames - Lw
                pad = np.clip(base[-1:], 0, max(Lw - 1, 0)).repeat(pad_needed)
                idxs = np.concatenate([base, pad], axis=0)
        frame_ids = item.t_start + idxs
        frame_ids = np.clip(frame_ids, 0, max(total_frames - 1, 0)).astype(int).tolist()
        frames = self._load_frames(item.source_path, frame_ids, item.video_id)
        tensor_frames = [self.transform(frame) for frame in frames]
        video = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sample = {
            "video": video,
            "label": torch.tensor(item.label, dtype=torch.long),
            "video_id": item.video_id,
        }
        audio = self._load_audio_clip(item.video_id, frame_ids)
        if audio is not None:
            sample["audio"] = audio
        return sample
