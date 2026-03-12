#!/usr/bin/env python3
"""Summarize Version-C B runs into a compact, upload-friendly artifact.

Outputs (small):
  - TSV: one row per run (enough to compute wins/loss/tie + meanΔ offline)
  - JSONL: same info, easier for programmatic parsing
  - MD: a few pairwise comparisons (overall + by budget)

Designed to be called by scripts/pack_B_outputs.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_EXP_KEYS = sorted(
    [
        "EXP-B2-bc2cec-probe-raw",
        "EXP-B2-bc2cec-noprobe",
        "EXP-B2-std-budgetaware",
        "EXP-B2-bc2cec",
        "EXP-B2-mpvs-only",
        "EXP-B1",
        "EXP-B3",
        "EXP-B2-naive-atomiconly",
        "EXP-B2-naive-relinkonly",
        "EXP-B2-naive-shakeonly",
        "EXP-B2-naive-tabuonly",
    ],
    key=len,
    reverse=True,
)


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _parse_budget_str(b: str) -> Tuple[str, int]:
    s = str(b or "").strip().lower()
    if not s:
        return "", 0
    if s.isdigit():
        return s, int(s)
    m = re.fullmatch(r"(\d+)([km])", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        return s, n * (1000 if unit == "k" else 1_000_000)
    return s, 0


def _parse_w_tag(s: str) -> float:
    s = str(s or "").strip().lower().replace("p", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def infer_exp_key(exp_dir_name: str) -> Tuple[str, str]:
    """Return (exp_key, run_tag_prefix)."""
    name = str(exp_dir_name)
    before = name.split("__", 1)[0]
    for k in KNOWN_EXP_KEYS:
        if before.startswith(k):
            rest = before[len(k) :].lstrip("-")
            return k, rest
    parts = before.split("-")
    exp_key = "-".join(parts[:3]) if len(parts) >= 3 else before
    rest = before[len(exp_key) :].lstrip("-")
    return exp_key, rest


_GRID_RE = re.compile(r"__b(?P<b>[^_]+)_wT(?P<wT>[^_]+)_wC(?P<wC>[^_]+)")
_INST_RE = re.compile(r"-(chain_skip_randw|chain_skip|cluster4)$")
_CM_RE = re.compile(r"(?:^|_)cm(?P<cm>[A-Z0-9_]+)(?:_|-|$)")
_V_RE = re.compile(r"V_E[0-3]_C[0-1]_D[0-3]_S[0-2]_K[0-1]_M[0-1]")


def parse_exp_dir_meta(exp_dir_name: str) -> Dict[str, Any]:
    exp_key, run_tag = infer_exp_key(exp_dir_name)
    name = str(exp_dir_name)
    inst = ""
    m_inst = _INST_RE.search(name)
    if m_inst:
        inst = str(m_inst.group(1) or "")
        name = name[: m_inst.start()]
    budget_str = ""
    wT = 0.0
    wC = 0.0
    m = _GRID_RE.search(name)
    if m:
        budget_str = m.group("b")
        wT = _parse_w_tag(m.group("wT"))
        wC = _parse_w_tag(m.group("wC"))
    cmode = ""
    m_cm = _CM_RE.search(str(exp_dir_name))
    if m_cm:
        cmode = str(m_cm.group("cm") or "").strip().lower()
    vtag = ""
    m_v = _V_RE.search(str(exp_dir_name))
    if m_v:
        vtag = str(m_v.group(0) or "")

    # If calls_mode isn't explicitly tagged as _cmEFF/_cmMISS, derive from VARIANT_TAG M-axis.
    if not cmode and vtag:
        if vtag.endswith("_M1"):
            cmode = "eff"
        elif vtag.endswith("_M0"):
            cmode = "miss"
    bstr_norm, b_calls = _parse_budget_str(budget_str)
    return {
        "exp_key": exp_key,
        "run_tag": run_tag,
        "budget": bstr_norm,
        "budget_calls_guess": int(b_calls),
        "wT": float(wT),
        "wC": float(wC),
        "instance": inst,
        "calls_mode": cmode,
        "variant_tag": vtag,
    }


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def compute_anytime(trace_csv: Path, total_budget_calls: int) -> Dict[str, Any]:
    if not trace_csv.exists() or total_budget_calls <= 0:
        return {}
    fracs = [0.25, 0.5, 0.75, 1.0]
    thresholds = [int(round(total_budget_calls * f)) for f in fracs]
    got = {t: False for t in thresholds}
    best = float("inf")
    last_calls = 0
    last_best = float("inf")
    auc = 0.0
    best_at: Dict[float, float] = {}
    final_total = None
    try:
        with trace_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                calls = _to_int(row.get("eval_calls_cum", 0), 0)
                tot = _to_float(row.get("total_scalar", float("inf")), float("inf"))
                if calls < last_calls:
                    continue
                if last_best != float("inf") and calls > last_calls:
                    auc += float(last_best) * float(calls - last_calls)
                if tot < best:
                    best = tot
                last_best = best
                last_calls = calls
                final_total = tot
                for t in thresholds:
                    if (not got[t]) and calls >= t:
                        best_at[float(t) / float(total_budget_calls)] = float(best)
                        got[t] = True
        if last_best != float("inf") and last_calls < total_budget_calls:
            auc += float(last_best) * float(total_budget_calls - last_calls)
    except Exception:
        return {}
    if not best_at:
        return {}
    return {
        "auc_best_total": float(auc) / float(max(1, total_budget_calls)),
        "best@0.25": best_at.get(0.25, None),
        "best@0.50": best_at.get(0.5, None),
        "best@0.75": best_at.get(0.75, None),
        "best@1.00": best_at.get(1.0, None),
        "final_total": float(final_total) if final_total is not None else None,
    }


def summarize_stage_probes(mpvs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mpvs or not isinstance(mpvs, dict):
        return {}
    sp = mpvs.get("stage_probes")
    if not isinstance(sp, list) or not sp:
        return {}
    n = 0
    atomic_eff: List[int] = []
    atomic_miss: List[int] = []
    cf_w: List[float] = []
    use_cf: List[int] = []
    raw_eff: List[int] = []
    raw_miss: List[int] = []
    raw_eq1 = 0
    for ev in sp:
        if not isinstance(ev, dict):
            continue
        n += 1
        atomic_eff.append(_to_int(ev.get("atomic_calls", 0), 0))
        acd = ev.get("atomic_calls_detail")
        if isinstance(acd, dict):
            atomic_miss.append(_to_int(acd.get("miss_calls", 0), 0))
        ops = ev.get("ops")
        if not isinstance(ops, list):
            continue
        for op in ops:
            if not isinstance(op, dict):
                continue
            cd = op.get("calls_detail")
            if isinstance(cd, dict):
                reff = _to_int(cd.get("eff_calls", op.get("raw_calls", 0)), 0)
                rmis = _to_int(cd.get("miss_calls", op.get("raw_calls", 0)), 0)
            else:
                reff = _to_int(op.get("raw_calls", 0), 0)
                rmis = reff
            raw_eff.append(reff)
            raw_miss.append(rmis)
            if reff == 1:
                raw_eq1 += 1
            cf_w.append(_to_float(op.get("cf_weight", 0.0), 0.0))
            use_cf.append(_to_int(op.get("use_cf", 0), 0))

    def _mean(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        return float(sum(xs)) / float(len(xs))

    return {
        "stage_probe_n": int(n),
        "atomic_calls_eff_mean": _mean([float(x) for x in atomic_eff]) if atomic_eff else None,
        "atomic_calls_miss_mean": _mean([float(x) for x in atomic_miss]) if atomic_miss else None,
        "raw_calls_eff_mean": _mean([float(x) for x in raw_eff]) if raw_eff else None,
        "raw_calls_miss_mean": _mean([float(x) for x in raw_miss]) if raw_miss else None,
        "raw_calls_eq1_ratio": float(raw_eq1) / float(max(1, len(raw_eff))) if raw_eff else None,
        "use_cf_ratio": float(sum(1 for x in use_cf if x)) / float(max(1, len(use_cf))) if use_cf else None,
        "cf_weight_mean": _mean(cf_w) if cf_w else None,
    }


def iter_run_dirs(run_dirs_file: Path) -> Iterable[Path]:
    with run_dirs_file.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            d = Path(p)
            if d.exists() and d.is_dir():
                yield d


def build_row(run_dir: Path) -> Dict[str, Any]:
    exp_dir = run_dir.parent.parent.name
    seed_dir = run_dir.parent.name
    run_id = run_dir.name
    meta = parse_exp_dir_meta(exp_dir)
    seed = _to_int(seed_dir.replace("seed", ""), -1)
    rs = load_json(run_dir / "run_summary.json") or {}
    budget_limit = _to_int(((rs.get("budget") or {}).get("primary_limit") or {}).get("limit", 0), 0)
    total_budget = budget_limit if budget_limit > 0 else _to_int(meta.get("budget_calls_guess", 0), 0)
    row: Dict[str, Any] = {
        "exp_key": meta["exp_key"],
        "run_tag": meta["run_tag"],
        "budget": meta["budget"],
        "budget_calls": int(total_budget),
        "wT": float(meta["wT"]),
        "wC": float(meta["wC"]),
        "instance": meta["instance"],
        "calls_mode": str(meta.get("calls_mode", "") or ""),
        "variant_tag": str(meta.get("variant_tag", "") or ""),
        "seed": int(seed),
        "run_id": str(run_id),
        "selected_total_scalar": _to_float(rs.get("selected_total_scalar", 0.0), 0.0),
        "budget_exhausted": int(bool(((rs.get("budget") or {}).get("budget_exhausted", False)))),
        "actual_eval_calls": _to_int(((rs.get("budget") or {}).get("actual_eval_calls", 0)), 0),
    }
    eff = rs.get("efficiency") or {}
    row.update(
        {
            "calls_per_iter_overall": _to_float(eff.get("calls_per_iter_overall", 0.0), 0.0),
            "steps_per_1k_calls_overall": _to_float(eff.get("steps_per_1k_calls_overall", 0.0), 0.0),
            "improve_steps_per_1k_calls_lastN": _to_float(eff.get("improve_steps_per_1k_calls_lastN", 0.0), 0.0),
            "unique_sigs_per_1k_calls_lastN": _to_float(eff.get("unique_sigs_per_1k_calls_lastN", 0.0), 0.0),
            "op_none_ratio_lastN": _to_float(eff.get("op_none_ratio_lastN", 0.0), 0.0),
        }
    )
    mpvs = rs.get("mpvs") if isinstance(rs.get("mpvs"), dict) else {}
    row.update(
        {
            "mpvs_enabled": int(bool(mpvs.get("enabled", False))) if isinstance(mpvs, dict) else 0,
            "controller_enabled": _to_int(mpvs.get("controller_enabled", 0), 0),
            "macro_selected": _to_int(mpvs.get("macro_selected", 0), 0),
            "macro_scored": _to_int(mpvs.get("macro_scored", 0), 0),
            "macro_precheck_allowed": _to_int(mpvs.get("macro_precheck_allowed", 0), 0),
            "macro_precheck_blocked": _to_int(mpvs.get("macro_precheck_blocked", 0), 0),
            "verifier_calls_spent": _to_int(mpvs.get("verifier_calls_spent", 0), 0),
        }
    )
    mops = mpvs.get("macro_ops") if isinstance(mpvs, dict) else {}
    if isinstance(mops, dict):
        for k in ("relink", "shake", "tabu_search"):
            st = mops.get(k) if isinstance(mops.get(k), dict) else {}
            row[f"{k}_tries"] = _to_int(st.get("tries", 0), 0)
            row[f"{k}_success"] = _to_int(st.get("success", 0), 0)
            row[f"{k}_fail"] = _to_int(st.get("fail", 0), 0)
            row[f"{k}_w"] = _to_float(st.get("weight", 0.0), 0.0)
            row[f"{k}_ewma_cf_rate"] = _to_float(st.get("ewma_cf_gain_per_call", 0.0), 0.0)
    row.update(summarize_stage_probes(mpvs))
    row.update(compute_anytime(run_dir / "trace.csv", total_budget_calls=int(total_budget)))
    return row


def write_tsv(rows: List[Dict[str, Any]], out_tsv: Path) -> None:
    if not rows:
        out_tsv.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_jsonl(rows: List[Dict[str, Any]], out_jsonl: Path) -> None:
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_pairwise(rows: List[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
    cell: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for r in rows:
        key = (
            r.get("run_tag"),
            _to_int(r.get("budget_calls", 0), 0),
            _to_float(r.get("wT", 0.0), 0.0),
            _to_float(r.get("wC", 0.0), 0.0),
            r.get("instance"),
            _to_int(r.get("seed", -1), -1),
        )
        ek = str(r.get("exp_key"))
        if ek not in {a, b}:
            continue
        cell.setdefault(key, {})[ek] = _to_float(r.get("selected_total_scalar", 0.0), 0.0)
    deltas: List[float] = []
    budgets: Dict[int, List[float]] = {}
    for k, v in cell.items():
        if a in v and b in v:
            d = float(v[a]) - float(v[b])
            deltas.append(d)
            budgets.setdefault(int(k[1]), []).append(d)

    def _stat(ds: List[float]) -> Dict[str, Any]:
        if not ds:
            return {"n": 0}
        wins = sum(1 for x in ds if x < 0)
        loss = sum(1 for x in ds if x > 0)
        tie = len(ds) - wins - loss
        mean = sum(ds) / float(len(ds))
        wmag = sum(-x for x in ds if x < 0) / float(max(1, wins))
        lmag = sum(x for x in ds if x > 0) / float(max(1, loss))
        return {
            "n": int(len(ds)),
            "mean_delta": float(mean),
            "wins": int(wins),
            "loss": int(loss),
            "tie": int(tie),
            "avg_win": float(wmag),
            "avg_loss": float(lmag),
            "max_loss": float(max([x for x in ds if x > 0], default=0.0)),
            "max_win": float(max([-x for x in ds if x < 0], default=0.0)),
        }

    return {
        "pair": f"{a} vs {b}",
        "overall": _stat(deltas),
        "by_budget": {str(k): _stat(v) for k, v in sorted(budgets.items(), key=lambda x: x[0])},
    }


def write_compare_md(compare: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# B sweep compare (compact)\n\n")
    for blk in compare:
        lines.append(f"## {blk['pair']}\n\n")
        ov = blk.get("overall", {})
        lines.append(
            f"overall: n={ov.get('n',0)} meanΔ={ov.get('mean_delta',0.0):.6g} "
            f"wins/loss/tie={ov.get('wins',0)}/{ov.get('loss',0)}/{ov.get('tie',0)} "
            f"avg_win/avg_loss={ov.get('avg_win',0.0):.3g}/{ov.get('avg_loss',0.0):.3g} "
            f"max_loss={ov.get('max_loss',0.0):.3g}\n\n"
        )
        byb = blk.get("by_budget", {})
        if byb:
            lines.append("| budget_calls | n | meanΔ | wins | loss | tie | max_loss |\n")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
            for b, st in byb.items():
                lines.append(
                    f"| {b} | {st.get('n',0)} | {st.get('mean_delta',0.0):.6g} | "
                    f"{st.get('wins',0)} | {st.get('loss',0)} | {st.get('tie',0)} | "
                    f"{st.get('max_loss',0.0):.3g} |\n"
                )
            lines.append("\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs-file", type=str, required=True)
    ap.add_argument("--out-tsv", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    ap.add_argument(
        "--compare-pairs",
        type=str,
        default="EXP-B2-bc2cec:EXP-B2-std-budgetaware,EXP-B2-bc2cec:EXP-B2-bc2cec-probe-raw,EXP-B1:EXP-B2-std-budgetaware,EXP-B3:EXP-B2-std-budgetaware",
        help="comma-separated pairs a:b",
    )
    args = ap.parse_args()
    run_dirs_file = Path(args.run_dirs_file)
    out_tsv = Path(args.out_tsv)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
    out_md = Path(args.out_md) if args.out_md else None
    rows: List[Dict[str, Any]] = []
    for d in iter_run_dirs(run_dirs_file):
        try:
            rows.append(build_row(d))
        except Exception:
            continue
    rows.sort(
        key=lambda r: (
            str(r.get("exp_key")),
            str(r.get("run_tag")),
            _to_int(r.get("budget_calls", 0), 0),
            str(r.get("instance")),
            _to_int(r.get("seed", -1), -1),
            str(r.get("run_id")),
        )
    )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(rows, out_tsv)
    if out_jsonl is not None:
        write_jsonl(rows, out_jsonl)
    compare: List[Dict[str, Any]] = []
    pairs = [p.strip() for p in str(args.compare_pairs).split(",") if p.strip()]
    for p2 in pairs:
        if ":" not in p2:
            continue
        a, b = p2.split(":", 1)
        compare.append(compute_pairwise(rows, a.strip(), b.strip()))
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        write_compare_md(compare, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
