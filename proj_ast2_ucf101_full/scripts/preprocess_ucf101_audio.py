"""Preprocess UCF101 audio into per-frame .npy features."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

import numpy as np


if importlib.util.find_spec("librosa") is not None:
    librosa = importlib.import_module("librosa")
else:
    librosa = None


def _load_waveform(video_path: Path, sample_rate: int) -> np.ndarray:
    if librosa is None:
        return np.array([])
    if not video_path.is_file():
        return np.array([])
    waveform, _ = librosa.load(video_path, sr=sample_rate, mono=True)
    return waveform


def _extract_mel(waveform: np.ndarray, sample_rate: int, n_mels: int, hop_length: int, win_length: int) -> np.ndarray:
    if librosa is None or waveform.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
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
        start = int(round(edges[idx]))
        end = int(round(edges[idx + 1]))
        if end <= start:
            pooled[idx] = mel[:, min(start, t_spec - 1)]
        else:
            pooled[idx] = mel[:, start:end].mean(axis=1)
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/ucf101")
    parser.add_argument("--frames-root", type=str, default="")
    parser.add_argument("--video-root", type=str, default="")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    args = parser.parse_args()

    root = Path(args.root)
    frames_root = Path(args.frames_root) if args.frames_root else (root / "frames")
    video_root = Path(args.video_root) if args.video_root else None
    audio_root = Path(args.audio_root) if args.audio_root else (root / "audio")
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
            if video_root:
                candidates = [
                    video_root / class_dir.name / f"{video_dir.name}.avi",
                    video_root / class_dir.name / f"{video_dir.name}.mp4",
                    video_root / class_dir.name / f"{video_dir.name}.mkv",
                ]
                for cand in candidates:
                    if cand.is_file():
                        video_path = cand
                        break
            waveform = _load_waveform(video_path, args.sample_rate) if video_path else np.array([])
            mel = _extract_mel(waveform, args.sample_rate, args.n_mels, args.hop_length, args.win_length)
            features = _pool_mel_to_frames(mel, num_frames)
            if waveform.size == 0:
                print(f"[WARN] audio unavailable for {video_dir.name}, saving zero features.")
                features = np.zeros((num_frames, args.n_mels), dtype=np.float32)
            np.save(out_path, features.astype(np.float32))
            print(f"[audio] saved {out_path}")


if __name__ == "__main__":
    main()
