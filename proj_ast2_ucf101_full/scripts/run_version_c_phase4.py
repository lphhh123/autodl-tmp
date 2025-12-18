"""Entry for Version-C Phase4 layout experiments."""
import argparse
from utils.config import load_config
from trainers.version_c_phase4_trainer import run_phase4


def main():
    parser = argparse.ArgumentParser(description="Version-C Phase4")
    parser.add_argument("--cfg", type=str, default="./configs/version_c_experiments/phase4_layout_ours.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    run_phase4(cfg)


if __name__ == "__main__":
    main()
