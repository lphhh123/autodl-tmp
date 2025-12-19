"""Run ms/mem proxy training/eval placeholder."""
import argparse
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/proxy_ms_mem.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    print("[proxy_ms_mem] loaded config", cfg)
    print("Proxy training/eval placeholder complete.")


if __name__ == "__main__":
    main()
