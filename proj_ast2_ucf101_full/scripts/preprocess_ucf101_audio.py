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


def _load_waveform(video_path: Optional[Path], sample_rate: int) -> np.ndarray:
    if librosa is None or video_path is None or not video_path.is_file():
        return np.array([])
    try:
        waveform, _ = librosa.load(video_path, sr=sample_rate, mono=True)
    except Exception:
        return np.array([])
    return waveform


def _extract_mel(waveform: np.ndarray, sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    if librosa is None or waveform.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    try:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
    except Exception:
        return np.zeros((n_mels, 1), dtype=np.float32)
    return mel_db.astype(np.float32)


def _pool_mel_to_frames(mel: np.ndarray, num_frames: int) -> np.ndarray:
    n_mels, t_spec = mel.shape
    if num_frames <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    boundaries = np.linspace(0, t_spec, num_frames + 1)
    pooled = np.zeros((num_frames, n_mels), dtype=np.float32)
    for idx in range(num_frames):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
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
    parser.add_argument("--raw-video-root", type=str, default="")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--frames-root", type=str, default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    frames_root = None
    audio_root = None
    raw_video_root = None
    sample_rate = args.sample_rate
    n_mels = args.n_mels
    n_fft = args.n_fft
    hop_length = args.hop_length

    if args.cfg:
        cfg = load_config(args.cfg)
        data_cfg = cfg.data
        frames_root = Path(getattr(data_cfg, "frames_root", "data/ucf101/frames"))
        audio_root = Path(getattr(data_cfg, "audio_root", "data/ucf101/audio"))
        raw_video_root_value = getattr(data_cfg, "raw_video_root", "")
        raw_video_root = Path(raw_video_root_value) if raw_video_root_value else None
        sample_rate = int(getattr(data_cfg, "sample_rate", sample_rate))
        n_mels = int(getattr(data_cfg, "n_mels", n_mels))

    if args.frames_root:
        frames_root = Path(args.frames_root)
    if args.audio_root:
        audio_root = Path(args.audio_root)
    if args.raw_video_root:
        raw_video_root = Path(args.raw_video_root)

    if frames_root is None:
        frames_root = Path("data/ucf101/frames")
    if audio_root is None:
        audio_root = Path("data/ucf101/audio")

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
            video_path = None
            if raw_video_root:
                candidate = raw_video_root / class_dir.name / f"{video_dir.name}.avi"
                if candidate.is_file():
                    video_path = candidate
            try:
                waveform = _load_waveform(video_path, sample_rate)
                mel = _extract_mel(waveform, sample_rate, n_mels, n_fft, hop_length)
                t_spec = mel.shape[1]
                frame_count = _get_video_frame_count(video_path)
                target_frames = frame_count if frame_count is not None else t_spec
                if target_frames <= 0:
                    target_frames = t_spec
                features = _pool_mel_to_frames(mel, target_frames)
                if waveform.size == 0 or t_spec == 0:
                    raise RuntimeError("audio decode failed")
            except Exception as exc:
                frame_count = _get_video_frame_count(video_path)
                if frame_count is None:
                    target_frames = mel.shape[1] if "mel" in locals() else 1
                else:
                    target_frames = frame_count
                target_frames = max(1, int(target_frames))
                print(f"[WARN] audio unavailable for {video_dir.name}: {exc}. Saving zeros.")
                features = np.zeros((target_frames, n_mels), dtype=np.float32)
            np.save(out_path, features.astype(np.float32))
            print(f"[audio] saved {out_path}")


if __name__ == "__main__":
    main()
