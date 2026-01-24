from __future__ import annotations

# NOTE: run_layout_suite is an orchestrator for batch runs, not the OneCommand entrypoint.
# It intentionally dispatches to the canonical module runners to avoid path-entry ambiguity.
# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import itertools
import subprocess
import time
from typing import List, Dict, Any

import yaml

from scripts.summarize_layout_runs import summarize_run, _write_summary, write_grouped


def _load_suite(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_runner(cfg_path: Path) -> str:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    baseline = cfg.get("baseline", {}) or {}
    method = str(baseline.get("method", "") or "").strip()
    name = str(baseline.get("name", "") or "").strip().lower()

    heuragenix_methods = {"llm_hh", "random_hh", "or_solver"}
    if method in heuragenix_methods:
        return "run_layout_heuragenix.py"
    if name.startswith("heuragenix"):
        return "run_layout_heuragenix.py"
    return "run_layout_agent.py"


def run_layout_suite(inputs: List[str], cfgs: List[str], seeds: List[int], out_root: Path, backtrack_window: int = 10) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout_input, cfg, seed in itertools.product(inputs, cfgs, seeds):
        method = Path(cfg).stem
        input_name = Path(layout_input).stem
        run_dir = out_root / method / input_name / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        try:
            cfg_path = Path(cfg)
            runner = _pick_runner(cfg_path)
            runner_mod = (
                "scripts.run_layout_heuragenix" if runner.endswith("run_layout_heuragenix.py") else "scripts.run_layout_agent"
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    runner_mod,
                    "--layout_input",
                    str(layout_input),
                    "--cfg",
                    str(cfg_path),
                    "--out_dir",
                    str(run_dir),
                    "--seed",
                    str(seed),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            pass
        wall_time = time.perf_counter() - start
        (run_dir / "wall_time.txt").write_text(f"{wall_time:.4f}", encoding="utf-8")
        rows.append(summarize_run(run_dir, backtrack_window=backtrack_window, wall_time_override=wall_time))

    if rows:
        rows.sort(key=lambda r: (r.get("method", ""), r.get("input", ""), r.get("seed", "")))
        summary_csv = out_root / "layout_suite_summary.csv"
        _write_summary(rows, summary_csv)
        write_grouped(rows, out_root)


def main():
    parser = argparse.ArgumentParser(description="Run layout suite experiments")
    parser.add_argument("--suite", type=str, help="YAML file describing inputs/cfgs/seeds", default=None)
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--cfgs", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--out_root", type=str, default="outputs/layout_suite")
    parser.add_argument("--backtrack_window", type=int, default=10)
    args = parser.parse_args()

    if args.suite:
        suite_cfg = _load_suite(Path(args.suite))
        inputs = suite_cfg.get("inputs", [])
        cfgs = suite_cfg.get("methods", suite_cfg.get("cfgs", []))
        seeds = suite_cfg.get("seeds", [])
    else:
        inputs = args.inputs
        cfgs = args.cfgs
        seeds = args.seeds

    if not inputs or not cfgs or not seeds:
        raise SystemExit("inputs, cfgs, and seeds must be provided")

    run_layout_suite(inputs, cfgs, seeds, Path(args.out_root), backtrack_window=int(args.backtrack_window))


if __name__ == "__main__":
    main()
