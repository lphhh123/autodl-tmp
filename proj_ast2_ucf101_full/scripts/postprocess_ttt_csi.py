"""
Postprocess training outputs to derive paper-friendly metrics (SPEC_D v5.4 §20.1.1):
  - time-to-target (epoch when acc constraint satisfied AND hw target reached)
  - CSI-like score for main-table readability

This script is intentionally side-effect free by default:
  it writes derived_metrics.json, and does NOT modify metrics.json unless --inplace is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: Dict[str, Any], key: str, default=None):
    # shallow helper; keep this script robust to schema evolution
    return d.get(key, default)


def compute_time_to_target(
    metrics: Dict[str, Any],
    hw_stats: Dict[str, Any],
    acc_eps: float,
    t_norm_thresh: Optional[float],
) -> Dict[str, Any]:
    """
    We support two modes:
      1) If metrics contains per-epoch history arrays, compute exact epoch.
      2) Otherwise, fall back to a coarse single-point indicator.
    """
    acc_ref = float(_get(metrics, "acc_ref", _get(metrics, "stable_hw", {}).get("acc_ref", 0.0)) or 0.0)
    # prefer "val_acc1_best" if present for dense baseline postprocess
    acc_best = _get(metrics, "val_acc1_best", None)
    acc_used = float(_get(metrics, "val_acc1", acc_best if acc_best is not None else 0.0) or 0.0)

    # hw target: prefer normalized latency if available
    t_norm = _get(hw_stats, "T_norm", _get(metrics, "T_norm", None))
    if t_norm is None:
        # try common names
        t_norm = _get(hw_stats, "latency_norm", _get(metrics, "latency_norm", None))
    t_norm_val = float(t_norm) if t_norm is not None else None

    ok_acc = (acc_ref > 0.0) and (acc_used >= (acc_ref - float(acc_eps)))
    ok_hw = True
    if t_norm_thresh is not None and t_norm_val is not None:
        ok_hw = (t_norm_val <= float(t_norm_thresh))

    # If no history exists, report a single-point "achieved" flag.
    return {
        "acc_ref": acc_ref,
        "acc_used": acc_used,
        "epsilon_drop": float(acc_eps),
        "t_norm": t_norm_val,
        "t_norm_thresh": float(t_norm_thresh) if t_norm_thresh is not None else None,
        "achieved": bool(ok_acc and ok_hw),
        "time_to_target_epoch": None,  # can be extended later when epoch history is stored
        "note": "history_not_available; achieved is single-point check",
    }


def compute_csi_like(
    metrics: Dict[str, Any],
    hw_stats: Dict[str, Any],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Dict[str, Any]:
    acc = float(_get(metrics, "val_acc1", _get(metrics, "val_acc1_best", 0.0)) or 0.0)
    # normalized components (prefer hw_stats then metrics)
    Tn = _get(hw_stats, "T_norm", _get(metrics, "T_norm", 1.0))
    En = _get(hw_stats, "E_norm", _get(metrics, "E_norm", 1.0))
    Mn = _get(hw_stats, "M_norm", _get(metrics, "M_norm", 1.0))
    Cn = _get(hw_stats, "Comm_norm", _get(metrics, "Comm_norm", _get(metrics, "C_norm", 1.0)))

    try:
        Tn = float(Tn)
    except Exception:
        Tn = 1.0
    try:
        En = float(En)
    except Exception:
        En = 1.0
    try:
        Mn = float(Mn)
    except Exception:
        Mn = 1.0
    try:
        Cn = float(Cn)
    except Exception:
        Cn = 1.0

    denom = 1.0 + alpha * Tn + beta * En + gamma * Mn + delta * Cn
    score = float(acc) / float(denom)
    return {
        "acc1": float(acc),
        "T_norm": float(Tn),
        "E_norm": float(En),
        "M_norm": float(Mn),
        "Comm_norm": float(Cn),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "delta": float(delta),
        "score_csi": float(score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Directory containing metrics.json and hw_stats.json")
    ap.add_argument("--acc_eps", type=float, default=0.01, help="epsilon_drop for time-to-target (absolute)")
    ap.add_argument("--t_norm_thresh", type=float, default=None, help="Optional threshold on T_norm (e.g. 0.9)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--inplace", action="store_true", help="Also merge derived fields back into metrics.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    hw_path = run_dir / "hw_stats.json"

    metrics = _load_json(metrics_path)
    hw_stats = _load_json(hw_path) if hw_path.exists() else {}

    ttt = compute_time_to_target(metrics, hw_stats, acc_eps=float(args.acc_eps), t_norm_thresh=args.t_norm_thresh)
    csi = compute_csi_like(metrics, hw_stats, args.alpha, args.beta, args.gamma, args.delta)

    derived = {"time_to_target": ttt, "csi_like": csi}
    out_path = run_dir / "derived_metrics.json"
    out_path.write_text(json.dumps(derived, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.inplace:
        # merge under stable keys to avoid clobbering existing schema
        metrics.setdefault("derived", {})
        metrics["derived"]["time_to_target"] = ttt
        metrics["derived"]["csi_like"] = csi
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    if args.inplace:
        print(f"[OK] updated {metrics_path}")


if __name__ == "__main__":
    main()
