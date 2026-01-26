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
from typing import Dict, List, Tuple

import numpy as np
import yaml

from hw_proxy.layer_hw_proxy import LayerHwProxy
from utils.config import load_config


def _resolve_proxy_config(cfg) -> Tuple[str, str]:
    hw_cfg = getattr(cfg, "hw", None)
    proxy_cfg = getattr(cfg, "proxy", None)
    gpu_yaml = ""
    weight_dir = ""
    if hw_cfg is not None:
        gpu_yaml = str(getattr(hw_cfg, "gpu_yaml", "") or "")
        weight_dir = str(getattr(hw_cfg, "proxy_weight_dir", "") or "")
    if not gpu_yaml and proxy_cfg is not None:
        gpu_yaml = str(getattr(proxy_cfg, "gpu_yaml", "") or "")
    if not weight_dir and proxy_cfg is not None:
        weight_dir = str(getattr(proxy_cfg, "weight_dir", "") or "")
    return gpu_yaml, weight_dir


def _load_device_map(gpu_yaml: str) -> Tuple[List[str], Dict[str, Dict]]:
    with open(gpu_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and "chip_types" in data:
        device_map = {entry["name"]: entry for entry in data["chip_types"] if "name" in entry}
        device_names = list(device_map.keys())
        return device_names, device_map
    if isinstance(data, dict):
        device_names = list(data.keys())
        return device_names, dict(data)
    raise RuntimeError(f"Unexpected gpu_yaml format in {gpu_yaml}")


def _build_sanity_layers(device_cfg: Dict) -> List[Dict]:
    base = {
        "keep_ratio": 1.0,
        "complexity_ratio": 1.0,
        "precision": 1.0,
        "device_cfg": device_cfg,
        "cfg": "sanity",
    }
    return [
        {
            **base,
            "layer_type": 1,
            "layer_kind": "attn",
            "flops": 1.2e9,
            "bytes": 6.0e6,
            "seq_len": 128,
            "L_patch": 128,
            "embed_dim": 256,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
        {
            **base,
            "layer_type": 2,
            "layer_kind": "mlp",
            "flops": 9.0e8,
            "bytes": 2.5e6,
            "seq_len": 196,
            "L_patch": 196,
            "embed_dim": 384,
            "num_heads": 6,
            "mlp_ratio": 3.0,
        },
        {
            **base,
            "layer_type": 0,
            "layer_kind": "patch_embed",
            "flops": 2.0e8,
            "bytes": 1.2e6,
            "seq_len": 64,
            "L_patch": 64,
            "embed_dim": 128,
            "num_heads": 2,
            "mlp_ratio": 2.0,
        },
    ]


def _check_predictions(device_name: str, pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    lat_ms = np.asarray(pred.get("lat_ms", []), dtype=np.float32)
    mem_mb = np.asarray(pred.get("mem_mb", []), dtype=np.float32)
    power_w = np.asarray(pred.get("power_w", []), dtype=np.float32)

    for key, arr in {"lat_ms": lat_ms, "mem_mb": mem_mb, "power_w": power_w}.items():
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"[ProxyCheckFailed] {device_name} {key} has non-finite values")
        if np.any(arr < 0):
            raise RuntimeError(f"[ProxyCheckFailed] {device_name} {key} has negative values")

    lat_sum = float(np.sum(lat_ms)) if lat_ms.size else 0.0
    mem_max = float(np.max(mem_mb)) if mem_mb.size else 0.0
    energy_mj = float(np.sum(power_w * lat_ms)) / 1000.0 if power_w.size else 0.0
    return {"lat_ms_sum": lat_sum, "mem_mb_max": mem_max, "energy_mj_sum": energy_mj}


def check_all_proxies(cfg, out_dir: Path) -> Dict[str, Dict]:
    gpu_yaml, weight_dir = _resolve_proxy_config(cfg)
    if not gpu_yaml:
        raise RuntimeError("Missing gpu_yaml: set cfg.hw.gpu_yaml or cfg.proxy.gpu_yaml")
    if not weight_dir:
        raise RuntimeError("Missing proxy_weight_dir: set cfg.hw.proxy_weight_dir or cfg.proxy.weight_dir")

    device_names, device_map = _load_device_map(gpu_yaml)
    if not device_names:
        raise RuntimeError(f"No chip_types found in {gpu_yaml}")

    root = Path(weight_dir)
    results: Dict[str, Dict] = {}
    for dev in device_names:
        ckpt_dir = root / dev
        required = ["proxy_ms.pt", "proxy_peak_mem_mb.pt", "proxy_energy_mj.pt"]
        missing = [name for name in required if not (ckpt_dir / name).is_file()]
        if missing:
            missing_list = ", ".join(missing)
            raise RuntimeError(
                f"[ProxyMissing] device={dev} missing [{missing_list}] under root={root} (dir={ckpt_dir})"
            )
        device_cfg = device_map.get(dev, {})
        proxy = LayerHwProxy(
            device_name=dev,
            gpu_yaml=gpu_yaml,
            weight_dir=str(root),
            run_ctx={"device": dev, "cfg": "sanity"},
        )
        layers_cfg = _build_sanity_layers(device_cfg)
        pred = proxy.predict_layers_batch(layers_cfg)
        summary = _check_predictions(dev, pred)
        results[dev] = {
            "ckpt_dir": str(ckpt_dir),
            "passed": True,
            **summary,
        }

    out_path = out_dir / "proxy_ckpt_check.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
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
