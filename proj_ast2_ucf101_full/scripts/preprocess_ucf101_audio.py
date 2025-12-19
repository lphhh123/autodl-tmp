"""Preprocess UCF101 audio into per-frame .npy features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def dummy_extract_audio_features(video_path: Path, num_frames: int, feat_dim: int) -> np.ndarray:
    """Placeholder audio preprocessing.

    Replace this stub with actual audio extraction (e.g., librosa) in production.
    """
    return np.zeros((num_frames, feat_dim), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/ucf101")
    parser.add_argument("--feat-dim", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.root)
    frames_root = root / "frames"
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    for class_dir in frames_root.iterdir():
        if not class_dir.is_dir():
            continue
        for video_dir in class_dir.iterdir():
            if not video_dir.is_dir():
                continue
            frame_files = sorted(p for p in video_dir.glob("*.jpg") if p.is_file())
            if not frame_files:
                continue
            out_path = audio_root / f"{video_dir.name}.npy"
            if out_path.exists():
                continue
            features = dummy_extract_audio_features(video_dir, len(frame_files), args.feat_dim)
            np.save(out_path, features)
            print(f"[audio] saved {out_path}")


if __name__ == "__main__":
    main()
