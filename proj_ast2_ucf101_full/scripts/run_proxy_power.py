"""Run power proxy training/eval placeholder."""
import argparse
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/proxy_power.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    print("[proxy_power] loaded config", cfg)
    print("Power proxy placeholder complete.")


if __name__ == "__main__":
    main()
