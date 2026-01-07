"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from utils.config import load_config
from trainer.trainer_single_device import train_single_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        cfg.train.seed = int(args.seed)
        cfg.training.seed = int(args.seed)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.train.out_dir = str(out_dir)
    train_single_device(cfg)


if __name__ == "__main__":
    main()
