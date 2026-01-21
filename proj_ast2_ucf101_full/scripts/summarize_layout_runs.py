from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional


def inverse_signature_from_sig(sig: str) -> str:
    if sig.startswith("swap:"):
        return sig
    if sig.startswith("rel:"):
        try:
            _, slot_part = sig.split(":", 1)
            slot_str, move = slot_part.split(":", 1)
            frm, to = move.split("->")
            return f"rel:{slot_str}:{to}->{frm}"
        except ValueError:
            return sig
    if sig.startswith("cl:"):
        return sig
    return sig


def compute_oscillation_rate(trace_path: Path, window: int = 10) -> float:
    if not trace_path.exists():
        return 0.0
    recent: deque[str] = deque(maxlen=window)
    backtrack = 0
    accepted = 0
    with trace_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("accepted", "0")) != "1":
                continue
            sig = row.get("signature", "")
            if not sig:
                continue
            accepted += 1
            for past in recent:
                if inverse_signature_from_sig(past) == sig:
                    backtrack += 1
                    break
            recent.append(sig)
    if accepted == 0:
        return 0.0
    return backtrack / float(accepted)


def _read_best_from_pareto(pareto_csv: Path) -> Dict[str, float]:
    best_total = math.inf
    best_comm = math.inf
    best_therm = math.inf
    if not pareto_csv.exists():
        return {"best_total": math.inf, "best_comm": math.inf, "best_therm": math.inf}
    with pareto_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_total = "total_scalar" in (reader.fieldnames or [])
        for row in reader:
            comm = float(row.get("comm_norm", math.inf))
            therm = float(row.get("therm_norm", math.inf))
            total = float(row.get("total_scalar", comm + therm)) if has_total else float(comm + therm)
            best_total = min(best_total, total)
            best_comm = min(best_comm, comm)
            best_therm = min(best_therm, therm)
    return {"best_total": best_total, "best_comm": best_comm, "best_therm": best_therm}


def _llm_stats(llm_usage_path: Path) -> Dict[str, float]:
    if not llm_usage_path.exists():
        return {"llm_ok_rate": 1.0, "fallback_rate": 0.0, "llm_calls": 0}
    total = 0
    ok = 0
    fallback = 0
    with llm_usage_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ok" in rec:
                total += 1
                if rec.get("ok", False):
                    ok += 1
                else:
                    fallback += 1
            elif rec.get("event") == "llm_step_failed" or rec.get("event") == "llm_init_failed":
                fallback += 1
    if total == 0:
        return {"llm_ok_rate": 1.0, "fallback_rate": float(fallback), "llm_calls": 0}
    return {
        "llm_ok_rate": ok / float(total),
        "fallback_rate": fallback / float(total),
        "llm_calls": total,
    }


def summarize_run(run_dir: Path, backtrack_window: int = 10, wall_time_override: Optional[float] = None) -> Dict[str, object]:
    pareto_stats = _read_best_from_pareto(run_dir / "pareto_points.csv")
    trace_path = run_dir / "trace.csv"
    oscillation_rate = compute_oscillation_rate(trace_path, window=backtrack_window)
    trace_lines = 0
    accepted_count = 0
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_lines += 1
                if str(row.get("accepted", "0")) == "1":
                    accepted_count += 1
    llm_stats = _llm_stats(run_dir / "llm_usage.jsonl")
    budget_path = run_dir / "budget.json"
    actual_eval_calls = None
    budget_main_axis = None
    if budget_path.exists():
        try:
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            actual_eval_calls = int(budget.get("actual_eval_calls", 0))
            budget_main_axis = budget.get("budget_main_axis")
        except Exception:
            actual_eval_calls = None
            budget_main_axis = None

    wall_time = wall_time_override
    wall_path = run_dir / "wall_time.txt"
    if wall_time is None and wall_path.exists():
        try:
            wall_time = float(wall_path.read_text().strip())
        except Exception:
            wall_time = None

    return {
        "run_dir": str(run_dir),
        "method": run_dir.parent.name,
        "input": run_dir.parent.parent.name if run_dir.parent.parent != run_dir.parent else run_dir.parent.name,
        "seed": run_dir.name,
        "best_total": pareto_stats.get("best_total", math.inf),
        "best_comm": pareto_stats.get("best_comm", math.inf),
        "best_therm": pareto_stats.get("best_therm", math.inf),
        "hypervolume": None,
        "n_eval": actual_eval_calls if actual_eval_calls is not None else trace_lines,
        "accepted": accepted_count,
        "oscillation_rate": oscillation_rate,
        "llm_ok_rate": llm_stats.get("llm_ok_rate", 1.0),
        "fallback_rate": llm_stats.get("fallback_rate", 0.0),
        "wall_time": wall_time,
        "budget_main_axis": budget_main_axis,
    }


def _write_summary(rows: List[Dict[str, object]], out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "input",
                "seed",
                "best_total",
                "best_comm",
                "best_therm",
                "hypervolume",
                "n_eval",
                "accepted",
                "oscillation_rate",
                "llm_ok_rate",
                "fallback_rate",
                "wall_time",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _agg_mean_std(vals: List[float]) -> str:
    if not vals:
        return ""
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.4g}"
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    return f"{mean:.4g}±{std:.3g}"


def write_grouped(rows: List[Dict[str, object]], out_dir: Path):
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)

    summary_rows = []
    for method, items in grouped.items():
        summary_rows.append(
            {
                "method": method,
                "best_total": _agg_mean_std([float(i.get("best_total", math.inf)) for i in items if math.isfinite(float(i.get("best_total", math.inf)))]),
                "hypervolume": _agg_mean_std([float(i.get("hypervolume", 0.0)) for i in items if i.get("hypervolume") is not None]),
                "wall_time": _agg_mean_std([float(i.get("wall_time")) for i in items if i.get("wall_time") is not None]),
                "oscillation_rate": _agg_mean_std([float(i.get("oscillation_rate", 0.0)) for i in items]),
                "llm_ok_rate": _agg_mean_std([float(i.get("llm_ok_rate", 0.0)) for i in items]),
            }
        )

    grouped_csv = out_dir / "summary_grouped.csv"
    grouped_csv.parent.mkdir(parents=True, exist_ok=True)
    with grouped_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(writer=f, fieldnames=["method", "best_total", "hypervolume", "wall_time", "oscillation_rate", "llm_ok_rate"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    md_lines = ["| method | best_total | hypervolume | wall_time | oscillation_rate | llm_ok_rate |", "|---|---|---|---|---|---|"]
    for row in summary_rows:
        md_lines.append(
            f"| {row['method']} | {row['best_total']} | {row['hypervolume']} | {row['wall_time']} | {row['oscillation_rate']} | {row['llm_ok_rate']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def summarize_directory(root: Path, backtrack_window: int = 10) -> List[Dict[str, object]]:
    run_dirs = [p.parent for p in root.rglob("report.json")]
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        rows.append(summarize_run(run_dir, backtrack_window=backtrack_window))
    rows.sort(key=lambda x: (x.get("method", ""), x.get("input", ""), x.get("seed", "")))
    if rows:
        budget_axes = {row.get("budget_main_axis") for row in rows if row.get("budget_main_axis") is not None}
        if len(budget_axes) > 1:
            raise RuntimeError(f"budget_main_axis mismatch across runs: {sorted(list(budget_axes))}")
        _write_summary(rows, root / "summary.csv")
        write_grouped(rows, root)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize layout runs")
    parser.add_argument("root", type=str, help="Directory containing run outputs")
    parser.add_argument("--backtrack_window", type=int, default=10)
    args = parser.parse_args()

    summarize_directory(Path(args.root), backtrack_window=int(args.backtrack_window))


if __name__ == "__main__":
    main()
