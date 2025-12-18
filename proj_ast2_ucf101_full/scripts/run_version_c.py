"""Entry for Version-C training."""
import argparse
from utils.config import load_config
from trainer.trainer_version_c import train_version_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.config or args.cfg or "./configs/vc_phase3_full_ucf101.yaml"
    cfg = load_config(cfg_path)
    train_version_c(cfg)


if __name__ == "__main__":
    main()
