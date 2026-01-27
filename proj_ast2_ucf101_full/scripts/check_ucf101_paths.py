"""Quick self-check for UCF101 dataset path resolution."""
from __future__ import annotations

import argparse
import sys

from utils.config import load_config
from utils.data_ucf101 import UCF101Dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Check UCF101 dataset path resolution.")
    parser.add_argument("--cfg", required=True, help="Path to config YAML.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.cfg)
        train_ds = UCF101Dataset(cfg, split="train")
        val_ds = UCF101Dataset(cfg, split="val")
    except Exception as exc:
        print("[UCF101Dataset CHECK] Failed to initialize dataset.")
        print(str(exc))
        return 1

    print("[UCF101Dataset CHECK] train frames_root:", train_ds.root)
    print("[UCF101Dataset CHECK] train splits_root:", train_ds.splits_root)
    print("[UCF101Dataset CHECK] train split_file:", train_ds.split_file)
    print("[UCF101Dataset CHECK] val frames_root:", val_ds.root)
    print("[UCF101Dataset CHECK] val splits_root:", val_ds.splits_root)
    print("[UCF101Dataset CHECK] val split_file:", val_ds.split_file)
    print("[UCF101Dataset CHECK] train len:", len(train_ds))
    print("[UCF101Dataset CHECK] val len:", len(val_ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
