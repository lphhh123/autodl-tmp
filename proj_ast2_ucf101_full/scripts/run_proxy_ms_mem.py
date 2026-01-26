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
from typing import Dict

import yaml

from hw_proxy.layer_hw_proxy import LayerHwProxy
from utils.config import load_config
from utils.proxy_ckpt_links import REQUIRED_FILES, ensure_proxy_ckpts_dir




def check_all_proxies(cfg, out_dir: Path) -> Dict[str, Dict]:
    project_root = Path(__file__).resolve().parents[1]

    root = Path(str(cfg.hw.proxy_weight_dir)).resolve()
    root.mkdir(parents=True, exist_ok=True)

    gpu_yaml = str(cfg.hw.gpu_yaml)
    with open(gpu_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    chip_types = y.get("chip_types", [])
    chip_map = {ct["name"]: ct for ct in chip_types if "name" in ct}
    device_names = [ct["name"] for ct in chip_types]

    results = {}

    for dev in device_names:
        dst_dir = root / dev
        ensure_proxy_ckpts_dir(project_root=project_root, device_name=dev, dst_dir=dst_dir)

        missing = [fn for fn in REQUIRED_FILES if not (dst_dir / fn).is_file()]
        if missing:
            raise RuntimeError(
                f"[ProxyMissingAfterMaterialize] device={dev} missing={missing} under dst_dir={dst_dir} root={root}"
            )

        proxy = LayerHwProxy(
            device_name=dev,
            gpu_yaml=gpu_yaml,
            weight_dir=str(root),
            run_ctx={"img": 224, "bs": 1},
        )

        layers = [
            {
                "layer_kind": "patch_embed",
                "flops": 1e9,
                "bytes": 5e8,
                "embed_dim": 768,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "seq_len": 196,
                "precision": 1.0,
                "keep_ratio": 1.0,
            },
            {
                "layer_kind": "attn",
                "flops": 3e9,
                "bytes": 2e9,
                "embed_dim": 768,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "seq_len": 196,
                "precision": 1.0,
                "keep_ratio": 1.0,
            },
            {
                "layer_kind": "mlp",
                "flops": 4e9,
                "bytes": 3e9,
                "embed_dim": 768,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "seq_len": 196,
                "precision": 1.0,
                "keep_ratio": 1.0,
            },
        ]
        pred = proxy.predict_layers_batch(layers)
        lat = pred["lat_ms"].tolist()
        mem = pred["mem_mb"].tolist()
        pw = pred["power_w"].tolist()

        def _ok(xs):
            return all((x == x) and (x >= 0.0) and (x < 1e12) for x in xs)

        if not (_ok(lat) and _ok(mem) and _ok(pw)):
            raise RuntimeError(f"[ProxySanityFail] device={dev} pred(lat,mem,power)={(lat, mem, pw)}")

        yaml_chip_entry = chip_map.get(dev, {})
        tdp = float(yaml_chip_entry.get("tdp_w", 300.0))
        for i, (ms, pw_i) in enumerate(zip(lat, pw)):
            if pw_i > 5.0 * tdp:
                implied_energy_mj = pw_i * ms
                print(
                    "[WARN][ProxyPowerOutOfRange] "
                    f"device={dev} layer#{i} power_w={pw_i:.3f} tdp_w={tdp:.1f} "
                    f"lat_ms={ms:.6f} implied_energy_mj={implied_energy_mj:.3f}"
                )

        results[dev] = {
            "ckpt_dir": str(dst_dir),
            "lat_ms": lat,
            "mem_mb": mem,
            "power_w": pw,
        }

    (out_dir / "proxy_ckpt_check.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print("[run_proxy_ms_mem] OK devices:", list(results.keys()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/proxy_ms_mem.yaml")
    parser.add_argument("--calib_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    if args.calib_csv or str(getattr(getattr(cfg, "hw", None), "calib_csv", "") or ""):
        print("[proxy_ms_mem][WARN] calib_csv is ignored; running proxy checkpoint sanity check only.")

    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/proxy_ms_mem") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else auto_out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = check_all_proxies(cfg, out_dir)
    print("[proxy_ms_mem] done:", results)


if __name__ == "__main__":
    main()
