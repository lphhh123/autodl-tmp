"""Preprocess UCF101 audio into per-frame .npy features."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np

from utils.config import load_config

if importlib.util.find_spec("librosa") is not None:
    librosa = importlib.import_module("librosa")
else:
    librosa = None

if importlib.util.find_spec("cv2") is not None:
    cv2 = importlib.import_module("cv2")
else:
    cv2 = None


def _load_waveform(video_path: Path, sample_rate: int) -> np.ndarray:
    if librosa is None:
        return np.array([])
    if not video_path.is_file():
        return np.array([])
    waveform, _ = librosa.load(video_path, sr=sample_rate, mono=True)
    return waveform


def _extract_mel(waveform: np.ndarray, sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    if librosa is None or waveform.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def _pool_mel_to_frames(mel: np.ndarray, num_frames: int) -> np.ndarray:
    n_mels, t_spec = mel.shape
    if num_frames <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    edges = np.linspace(0, t_spec, num_frames + 1)
    pooled = np.zeros((num_frames, n_mels), dtype=np.float32)
    for idx in range(num_frames):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end <= start:
            end = start + 1
        end = min(end, t_spec)
        start = min(start, max(t_spec - 1, 0))
        pooled[idx] = mel[:, start:end].mean(axis=1)
    return pooled


def _get_video_frame_count(video_path: Optional[Path]) -> Optional[int]:
    if video_path is None or not video_path.is_file() or cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--root", type=str, default="data/ucf101")
    parser.add_argument("--frames-root", type=str, default="")
    parser.add_argument("--raw-video-root", type=str, default="")
    parser.add_argument("--video-root", type=str, default="")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    frames_root = None
    audio_root = None
    raw_video_root = None

    if args.cfg:
        cfg = load_config(args.cfg)
        data_cfg = cfg.data
        frames_root = Path(getattr(data_cfg, "frames_root", "data/ucf101/frames"))
        audio_root = Path(getattr(data_cfg, "audio_root", "data/ucf101/audio"))
        raw_video_root_value = getattr(data_cfg, "raw_video_root", "")
        raw_video_root = Path(raw_video_root_value) if raw_video_root_value else None

    if args.frames_root:
        frames_root = Path(args.frames_root)
    if args.audio_root:
        audio_root = Path(args.audio_root)
    if args.raw_video_root:
        raw_video_root = Path(args.raw_video_root)
    if args.video_root and raw_video_root is None:
        raw_video_root = Path(args.video_root)

    root = Path(args.root)
    if frames_root is None:
        frames_root = root / "frames"
    if audio_root is None:
        audio_root = root / "audio"
    if not frames_root.is_absolute():
        frames_root = project_root / frames_root
    if not audio_root.is_absolute():
        audio_root = project_root / audio_root
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
            num_frames = len(frame_files)
            video_path = None
            if raw_video_root:
                candidates = [
                    raw_video_root / class_dir.name / f"{video_dir.name}.avi",
                    raw_video_root / class_dir.name / f"{video_dir.name}.mp4",
                    raw_video_root / class_dir.name / f"{video_dir.name}.mkv",
                ]
                for cand in candidates:
                    if cand.is_file():
                        video_path = cand
                        break
            waveform = _load_waveform(video_path, args.sample_rate) if video_path else np.array([])
            mel = _extract_mel(waveform, args.sample_rate, args.n_mels, args.n_fft, args.hop_length)
            frame_count = _get_video_frame_count(video_path) if video_path else None
            target_frames = frame_count if frame_count is not None else mel.shape[1]
            if target_frames <= 0:
                target_frames = num_frames
            features = _pool_mel_to_frames(mel, target_frames)
            if waveform.size == 0:
                print(f"[WARN] audio unavailable for {video_dir.name}, saving zero features.")
                features = np.zeros((target_frames, args.n_mels), dtype=np.float32)
            np.save(out_path, features.astype(np.float32))
            print(f"[audio] saved {out_path}")


if __name__ == "__main__":
    main()
