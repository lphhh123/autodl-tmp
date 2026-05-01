from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_INSTANCES = ("chain_skip", "chain_skip_randw", "cluster4", "base")


def _budget_to_int(b: str) -> int:
    b = b.strip().lower()
    if b.endswith("k"):
        return int(b[:-1]) * 1000
    if b.endswith("m"):
        return int(b[:-1]) * 1_000_000
    return int(b)


def _wtwc_to_float(s: str) -> float:
    # "0p3" -> 0.3
    s = s.strip().lower().replace("p", ".")
    return float(s)


TAG_RE = re.compile(r"-b(?P<budget>[0-9]+[km]?)_wT(?P<wt>[0-9p]+)_wC(?P<wc>[0-9p]+)")


def _parse_exp_dir_name(name: str) -> Tuple[str, str, Optional[str], Optional[float], Optional[float]]:
    """Parse exp_dir basename.

    Returns:
      (exp_prefix, instance, budget_str, wT, wC)
    """
    inst = "base"
    base = name
    for cand in KNOWN_INSTANCES:
        suf = f"-{cand}"
        if name.endswith(suf):
            inst = cand
            base = name[: -len(suf)]
            break

    base = base.rstrip("_-")
    m = TAG_RE.search(base)
    if not m:
        return base, inst, None, None, None

    budget = m.group("budget")
    wt = _wtwc_to_float(m.group("wt"))
    wc = _wtwc_to_float(m.group("wc"))
    exp_prefix = base[: m.start()].rstrip("_-")
    return exp_prefix, inst, budget, wt, wc


def _is_budget_full(run_dir: Path) -> bool:
    p = run_dir / "budget.json"
    if not p.exists():
        return False
    try:
        b = json.loads(p.read_text(encoding="utf-8"))
        exhausted = bool(b.get("budget_exhausted", False))
        actual = int(b.get("actual_eval_calls", -1))
        lim = int(((b.get("primary_limit") or {}).get("limit", -1)))
        return exhausted and lim > 0 and actual == lim
    except Exception:
        return False


def _best_run_dir(seed_dir: Path) -> Optional[Path]:
    if not seed_dir.exists():
        return None
    run_dirs = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # pass 1: budget-full
    for d in run_dirs:
        if (d / "manifest.json").exists() and _is_budget_full(d):
            return d
    # pass 2: latest manifest
    for d in run_dirs:
        if (d / "manifest.json").exists():
            return d
    return None


def _safe_read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_runs(root: Path) -> Iterable[Dict[str, Any]]:
    for exp_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("EXP-")]):
        exp_prefix, inst, budget, wt, wc = _parse_exp_dir_name(exp_dir.name)

        for seed_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]):
            seed = seed_dir.name.replace("seed", "")
            best = _best_run_dir(seed_dir)
            if not best:
                continue
            report = _safe_read_json(best / "report.json")
            summ = _safe_read_json(best / "run_summary.json")

            selected = report.get("selected_total_scalar", None)
            try:
                selected_f = float(selected) if selected is not None else math.inf
            except Exception:
                selected_f = math.inf

            mpvs = (summ.get("mpvs") or {})
            macro_ops = mpvs.get("macro_ops") or {}

            row: Dict[str, Any] = {
                "exp_dir": exp_dir.name,
                "exp_prefix": exp_prefix,
                "instance": inst,
                "seed": int(seed) if seed.isdigit() else seed,
                "budget": budget,
                "budget_int": _budget_to_int(budget) if budget else None,
                "wT": wt,
                "wC": wc,
                "selected_total_scalar": selected_f,
                "run_id": best.name,
                "run_dir": str(best),
                "controller_enabled": mpvs.get("controller_enabled"),
                "macro_selected": mpvs.get("macro_selected"),
                "memory_selected": mpvs.get("memory_selected"),
                "macro_scored": mpvs.get("macro_scored"),
                "mem_scored": mpvs.get("mem_scored"),
                "nonheur_current_gate_blocked": mpvs.get("nonheur_current_gate_blocked"),
                "macro_precheck_fail_min_gain": mpvs.get("macro_precheck_fail_min_gain"),
            }

            for op_name, st in macro_ops.items():
                if not isinstance(st, dict):
                    continue
                for k in ("tries", "success", "fail", "ewma_gain_per_call", "cooldown", "weight"):
                    if k in st:
                        row[f"macroop_{op_name}_{k}"] = st.get(k)

            yield row


def _write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Analyze B runs: headroom oracle + regret + macro utilization")
    ap.add_argument("--root", type=str, default="outputs/B", help="B output root")
    ap.add_argument("--out_dir", type=str, default="outputs/B/_analysis", help="Where to write analysis files")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = list(_iter_runs(root)) if root.exists() else []
    runs.sort(
        key=lambda r: (
            str(r.get("instance")),
            int(r.get("budget_int") or 0),
            str(r.get("exp_prefix")),
            str(r.get("seed")),
        )
    )
    _write_csv(runs, out_dir / "run_table.csv")

    # Headroom probes (controller=0) used to build an oracle upper bound.
    # Keep legacy arms for backward compatibility; mainline uses the 3 enhancement components.
    headroom_arms = {
        "EXP-B2-naive-atomiconly": "atom",
        "EXP-B2-naive-relinkonly": "relink",
        "EXP-B2-naive-shakeonly": "shake",
        "EXP-B2-naive-tabuonly": "tabu",
        # legacy (optional)
        "EXP-B2-naive-chainonly": "chain",
        "EXP-B2-naive-ruinonly": "ruin",
        "EXP-B2-naive-macroonly": "macro",
        "EXP-B2-naive-blockonly": "block",
    }

    arm_order = ["atom", "relink", "shake", "tabu", "chain", "ruin", "block", "macro"]

    key_fields = ("instance", "budget", "wT", "wC", "seed")
    arm_by_key: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(dict)
    for r in runs:
        arm = headroom_arms.get(str(r.get("exp_prefix")), None)
        if not arm:
            continue
        k = tuple(r.get(f) for f in key_fields)
        arm_by_key[k][arm] = float(r.get("selected_total_scalar", math.inf))

    oracle_rows: List[Dict[str, Any]] = []
    oracle_best: Dict[Tuple[Any, ...], float] = {}
    for k, arms in arm_by_key.items():
        best_arm = None
        best_val = math.inf
        for a, v in arms.items():
            if v < best_val:
                best_val = v
                best_arm = a
        row = {f: v for f, v in zip(key_fields, k)}
        row.update({f"arm_{a}": arms.get(a) for a in arm_order})
        row["oracle_best"] = best_val if math.isfinite(best_val) else None
        row["oracle_arm"] = best_arm
        oracle_rows.append(row)
        oracle_best[k] = best_val
    oracle_rows.sort(
        key=lambda r: (
            str(r.get("instance")),
            _budget_to_int(str(r.get("budget"))) if r.get("budget") else 0,
            str(r.get("seed")),
        )
    )
    _write_csv(oracle_rows, out_dir / "oracle_arm.csv")

    main_methods = [
        "EXP-B1",
        "EXP-B2-mpvs-only",
        "EXP-B2-std-budgetaware",
        "EXP-B2-taos-style",
        "EXP-B2-bc2cec",
        "EXP-B2-bc2cec-nolong",
        "EXP-B2-bc2cec-noprobe",
        "EXP-B2-bc2cec-probe-raw",
        "EXP-B3",
    ]

    regret_rows: List[Dict[str, Any]] = []
    for r in runs:
        exp = str(r.get("exp_prefix"))
        if exp not in main_methods:
            continue
        k = tuple(r.get(f) for f in key_fields)
        o = oracle_best.get(k)
        if o is None or not math.isfinite(o):
            continue
        val = float(r.get("selected_total_scalar", math.inf))
        reg = val - o
        row = {f: r.get(f) for f in key_fields}
        row.update(
            {
                "method": exp,
                "selected_total_scalar": val,
                "oracle_best": o,
                "regret": reg,
                "controller_enabled": r.get("controller_enabled"),
                "macro_selected": r.get("macro_selected"),
                "memory_selected": r.get("memory_selected"),
                "nonheur_current_gate_blocked": r.get("nonheur_current_gate_blocked"),
                "macro_precheck_fail_min_gain": r.get("macro_precheck_fail_min_gain"),
                "run_dir": r.get("run_dir"),
            }
        )
        for kk, vv in r.items():
            if str(kk).startswith("macroop_") and str(kk).endswith("_tries"):
                row[kk] = vv
        regret_rows.append(row)

    regret_rows.sort(
        key=lambda r: (
            str(r.get("instance")),
            _budget_to_int(str(r.get("budget"))) if r.get("budget") else 0,
            str(r.get("method")),
            str(r.get("seed")),
        )
    )
    _write_csv(regret_rows, out_dir / "regret.csv")

    agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in regret_rows:
        try:
            agg[(str(r.get("method")), str(r.get("budget")))].append(float(r.get("regret")))
        except Exception:
            pass

    summary_rows = []
    for (m, b), vals in sorted(agg.items(), key=lambda x: (x[0][0], _budget_to_int(x[0][1]))):
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        summary_rows.append({"method": m, "budget": b, "mean_regret": mean, "n": len(vals)})
    _write_csv(summary_rows, out_dir / "regret_summary.csv")

    md = [
        "# B Oracle/Regret Summary",
        "",
        "Oracle: best among headroom arms {atom, relink, shake, tabu} under controller=0 (legacy arms also supported).",
        "Regret: selected_total_scalar(method) - oracle_best (lower is better).",
        "",
        "| method | budget | mean_regret | n |",
        "|---|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(f"| {r['method']} | {r['budget']} | {r['mean_regret']:.6g} | {r['n']} |")
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
