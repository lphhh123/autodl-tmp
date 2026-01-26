# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import time

from utils.config import load_config

from scripts.run_proxy_ms_mem import check_all_proxies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/proxy_power.yaml")
    parser.add_argument("--calib_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    if args.calib_csv or str(getattr(getattr(cfg, "hw", None), "calib_csv", "") or ""):
        print("[proxy_power][WARN] calib_csv is ignored; running proxy checkpoint sanity check only.")

    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/proxy_power") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else auto_out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = check_all_proxies(cfg, out_dir)
    print("[proxy_power] done:", results)


if __name__ == "__main__":
    main()
