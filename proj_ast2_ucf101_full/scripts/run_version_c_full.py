"""Entry for full Version-C training."""
import argparse
from utils.config import load_config
from trainers.version_c_full_trainer import train_version_c_full


def main():
    parser = argparse.ArgumentParser(description="Version-C full training")
    parser.add_argument("--cfg", type=str, default="./configs/version_c_ucf101.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    train_version_c_full(cfg)


if __name__ == "__main__":
    main()
