"""Entry for Version-C full training."""
import argparse
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch

from utils.config import load_config
from trainer.trainer_version_c import train_version_c


def _str2bool(value: Union[str, bool]) -> bool:
    """Parse flexible boolean flags.

    Accepts common textual forms (true/false/yes/no/1/0) so users can run either
    `--export_layout_input` or `--export_layout_input true` without argparse
    rejecting the value.
    """

    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"true", "t", "yes", "y", "1"}:
        return True
    if value_lower in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected for export_layout_input, got {value!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/version_c_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--export_layout_input",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Export layout_input.json per SPEC v4.3.2 (accepts flag alone or true/false)",
    )
    parser.add_argument("--export_dir", type=str, default=None, help="Directory to write layout artifacts")
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
    export_layout_input = args.export_layout_input or args.out_dir is not None
    export_dir = args.export_dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.train.out_dir = str(out_dir)
        if export_dir is None:
            export_dir = str(out_dir)
    train_version_c(cfg, export_layout_input=export_layout_input, export_dir=export_dir)


if __name__ == "__main__":
    main()
