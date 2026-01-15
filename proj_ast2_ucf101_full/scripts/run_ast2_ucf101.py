"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
from pathlib import Path

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from trainer.trainer_single_device import train_single_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="ast2")
    seed = int(args.seed)
    seed_everything(seed)
    if hasattr(cfg, "train"):
        cfg.train.seed = seed
    if hasattr(cfg, "training"):
        cfg.training.seed = seed
    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(cfg, "train"):
            cfg.train.out_dir = str(out_dir)
        with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
            f.write(Path(args.cfg).read_text(encoding="utf-8"))
    train_single_device(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
