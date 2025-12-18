"""Entry for Version-C Phase3 mapping baseline."""
import argparse
from utils.config import load_config
from trainers.version_c_phase3_trainer import run_phase3


def main():
    parser = argparse.ArgumentParser(description="Version-C Phase3")
    parser.add_argument("--cfg", type=str, default="./configs/version_c_experiments/phase3_mapping_baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    run_phase3(cfg)


if __name__ == "__main__":
    main()
