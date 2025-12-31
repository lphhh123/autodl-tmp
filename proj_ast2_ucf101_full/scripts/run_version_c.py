"""Entry for Version-C full training."""
import argparse
from utils.config import load_config
from trainer.trainer_version_c import train_version_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/version_c_ucf101.yaml")
    parser.add_argument("--export_layout_input", action="store_true", help="Export layout_input.json per SPEC v4.3.2")
    parser.add_argument("--export_dir", type=str, default=None, help="Directory to write layout artifacts")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    train_version_c(cfg, export_layout_input=args.export_layout_input, export_dir=args.export_dir)


if __name__ == "__main__":
    main()
