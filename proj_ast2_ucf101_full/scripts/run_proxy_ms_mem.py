# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import json
import time

from utils.config import load_config
from hw_proxy.proxy_train import train_layer_proxies_from_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/proxy_ms_mem.yaml")
    parser.add_argument("--calib_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/proxy_ms_mem") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else auto_out
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_csv = args.calib_csv or str(getattr(getattr(cfg, "hw", None), "calib_csv", "") or "")
    if not calib_csv:
        raise RuntimeError("Missing calib_csv: please provide --calib_csv or set cfg.hw.calib_csv")

    weight_dir = str(getattr(getattr(cfg, "hw", None), "proxy_weight_dir", "") or "")
    if not weight_dir:
        raise RuntimeError("Missing cfg.hw.proxy_weight_dir (where to save latency/mem/power proxy weights)")
    Path(weight_dir).mkdir(parents=True, exist_ok=True)

    meta = train_layer_proxies_from_csv(
        calib_csv=calib_csv,
        out_dir=weight_dir,
        in_dim=None,
        device="cpu",
        epochs=int(getattr(getattr(cfg, "hw", None), "proxy_train_epochs", 200) or 200),
        lr=float(getattr(getattr(cfg, "hw", None), "proxy_train_lr", 1e-3) or 1e-3),
    )

    (out_dir / "proxy_train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[proxy_ms_mem] done:", meta)


if __name__ == "__main__":
    main()
