"""Entry for Version-C full training."""
import argparse
from utils.config import load_config
from trainer.trainer_version_c import train_version_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/version_c_ucf101.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    train_version_c(cfg)


if __name__ == "__main__":
    main()
