
import os
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_video(
    video_path: Path,
    out_dir: Path,
    img_tmpl: str = "img_{:05d}.jpg",
    fps: int | None = None,
):
    """
    video_path: 原始 .avi / .mp4 路径
    out_dir:    输出帧路径目录，如 data/ucf101/frames/ApplyEyeMakeup/v_xxx_xxx/
    fps:        如果为 None，就用原始 FPS；否则按给定 FPS 采样（简单跳帧实现）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] failed to open video: {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = 1
    if fps is not None and fps > 0 and orig_fps > 0:
        frame_interval = max(int(round(orig_fps / fps)), 1)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            img_name = img_tmpl.format(saved + 1)
            img_path = out_dir / img_name
            cv2.imwrite(str(img_path), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[ok] {video_path} -> {out_dir}, saved {saved} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="UCF-101 原始视频根目录，例如 /data/UCF-101",
    )
    ap.add_argument(
        "--frame_root",
        type=str,
        required=True,
        help="输出帧根目录，例如 proj_ast2_ucf101_full/data/ucf101/frames",
    )
    ap.add_argument(
        "--ext",
        type=str,
        default=".avi",
        help="视频后缀，默认为 .avi，可改为 .mp4",
    )
    ap.add_argument(
        "--target_fps",
        type=int,
        default=None,
        help="如果想统一采样 fps，可以设定，例如 25；None 则用原始 fps",
    )
    args = ap.parse_args()

    video_root = Path(args.video_root)
    frame_root = Path(args.frame_root)
    ext = args.ext.lower()

    videos = []
    for cls_dir in sorted(video_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for vid_path in sorted(cls_dir.glob(f"*{ext}")):
            videos.append(vid_path)

    print(f"[info] found {len(videos)} videos under {video_root}")

    for vid in tqdm(videos):
        # 类名 = 上一级目录名
        cls_name = vid.parent.name
        # 视频名（去掉后缀），例如 v_ApplyEyeMakeup_g01_c01
        vid_name = vid.stem

        out_dir = frame_root / cls_name / vid_name
        extract_video(vid, out_dir, fps=args.target_fps)


if __name__ == "__main__":
    main()
