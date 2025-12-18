"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
from utils.config import load_config
from trainer.trainer_single_device import train_single_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    train_single_device(cfg)


if __name__ == "__main__":
    main()
