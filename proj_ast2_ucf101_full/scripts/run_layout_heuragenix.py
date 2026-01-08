"""HeurAgenix baseline wrapper for wafer layout (SPEC v4.3.2/4)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from layout.evaluator import LayoutEvaluator, LayoutState
from scripts.run_layout_agent import _compute_trace_metrics
from utils.config import load_config


def _load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_everything(seed: int) -> random.Random:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    return rng


def _resolve_heuragenix_root(cfg_root: str | None, project_root: Path) -> Path:
    if cfg_root:
        root = Path(cfg_root).expanduser()
        if root.exists():
            return root
    return project_root / "HeurAgenix"


def _write_instance(layout_input: dict, heuragenix_root: Path, out_dir: Path) -> Path:
    data_dir = heuragenix_root / "data" / "wafer_layout" / "data" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    instance_path = data_dir / f"instance_{out_dir.name}.json"
    with instance_path.open("w", encoding="utf-8") as f:
        json.dump(layout_input, f, indent=2)
    return instance_path


def _load_llm_config(heuragenix_root: Path, cfg_path: str | None) -> Dict[str, Any]:
    if not cfg_path:
        return {}
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = heuragenix_root / cfg_file
    if not cfg_file.exists():
        return {}
    with cfg_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_llm_env(llm_cfg: Dict[str, Any]) -> None:
    if "endpoint" in llm_cfg:
        os.environ.setdefault("VOLC_ARK_ENDPOINT", str(llm_cfg["endpoint"]))
    if "model" in llm_cfg:
        os.environ.setdefault("VOLC_ARK_MODEL", str(llm_cfg["model"]))
    if "api_key" in llm_cfg:
        os.environ.setdefault("VOLC_ARK_API_KEY", str(llm_cfg["api_key"]))


def _build_evaluator(cfg, layout_input: dict) -> LayoutEvaluator:
    return LayoutEvaluator(
        sigma_mm=float(cfg.objective.sigma_mm),
        baseline={
            "L_comm_baseline": float(layout_input["baseline"].get("L_comm", 1.0)),
            "L_therm_baseline": float(layout_input["baseline"].get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(cfg.objective.scalar_weights.w_comm),
            "w_therm": float(cfg.objective.scalar_weights.w_therm),
            "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
        },
    )


def _build_state(layout_input: dict, assign: np.ndarray) -> LayoutState:
    return LayoutState(
        S=int(layout_input["slots"]["S"]),
        Ns=int(layout_input["sites"]["Ns"]),
        wafer_radius_mm=float(layout_input["wafer"]["radius_mm"]),
        sites_xy_mm=np.asarray(layout_input["sites"]["sites_xy"], dtype=np.float32),
        assign=assign,
        chip_tdp_w=np.asarray(layout_input["slots"]["tdp"], dtype=float),
        traffic_bytes=np.asarray(layout_input["mapping"]["traffic_matrix"], dtype=float),
        meta={},
    )


def _action_from_operator(operator) -> Dict[str, Any]:
    name = getattr(operator, "name", operator.__class__.__name__).lower()
    if name == "swap":
        return {"op": "swap", "i": int(operator.i), "j": int(operator.j)}
    if name == "relocate":
        return {"op": "relocate", "i": int(operator.i), "site_id": int(operator.site_id)}
    if name == "cluster_move":
        return {"op": "cluster_move", "slots": list(operator.slots), "target_sites": list(operator.target_sites)}
    if name == "random_kick":
        return {"op": "random_kick", "ops": [getattr(op, "name", op.__class__.__name__) for op in operator.ops]}
    if name == "noop":
        return {"op": "none"}
    return {"op": name}


def _apply_action(assign: np.ndarray, action: Dict[str, Any]) -> np.ndarray:
    new_assign = assign.copy()
    op = action.get("op")
    if op == "swap":
        i = int(action["i"])
        j = int(action["j"])
        new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
    elif op == "relocate":
        i = int(action["i"])
        new_assign[i] = int(action["site_id"])
    elif op == "cluster_move":
        slots = [int(s) for s in action.get("slots", [])]
        sites = [int(s) for s in action.get("target_sites", [])]
        for slot, site in zip(slots, sites):
            new_assign[slot] = site
    return new_assign


def _assign_signature(assign: np.ndarray) -> str:
    return "assign:" + ",".join(str(int(a)) for a in assign.tolist())


def _write_usage_line(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    layout_input = _load_layout_input(Path(args.layout_input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_usage_path = out_dir / "llm_usage.jsonl"
    llm_usage_path.touch(exist_ok=True)

    rng = _seed_everything(int(args.seed))
    project_root = Path(__file__).resolve().parents[1]
    baseline_cfg = cfg.baseline if hasattr(cfg, "baseline") else {}
    heuragenix_root = _resolve_heuragenix_root(baseline_cfg.get("heuragenix_root"), project_root)

    instance_path = _write_instance(layout_input, heuragenix_root, out_dir)
    llm_cfg = _load_llm_config(heuragenix_root, baseline_cfg.get("llm_config_file"))
    _apply_llm_env(llm_cfg)

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(heuragenix_root))
    sys.path.insert(0, str(heuragenix_root / "src"))

    from problems.wafer_layout.env import Env
    from problems.wafer_layout.heuristics.basic_heuristics.best_swap_by_total_delta import (
        best_swap_by_total_delta,
    )
    from problems.wafer_layout.heuristics.basic_heuristics.best_relocate_to_free_by_total_delta import (
        best_relocate_to_free_by_total_delta,
    )
    from problems.wafer_layout.heuristics.basic_heuristics.comm_driven_swap import comm_driven_swap
    from problems.wafer_layout.heuristics.basic_heuristics.therm_driven_push_apart import therm_driven_push_apart
    from problems.wafer_layout.heuristics.basic_heuristics.region_shuffle_small import region_shuffle_small
    from problems.wafer_layout.heuristics.basic_heuristics.random_swap import random_swap
    from problems.wafer_layout.heuristics.basic_heuristics.random_relocate import random_relocate
    from problems.wafer_layout.heuristics.basic_heuristics.do_nothing import do_nothing
    from pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
    from pipeline.hyper_heuristics.random import RandomHyperHeuristic

    heuristics = [
        best_swap_by_total_delta,
        best_relocate_to_free_by_total_delta,
        comm_driven_swap,
        therm_driven_push_apart,
        region_shuffle_small,
        random_swap,
        random_relocate,
        do_nothing,
    ]

    env = Env(str(instance_path), rng=rng)
    initial_assign = np.asarray(env.current_solution.assign, dtype=int)

    method = str(baseline_cfg.get("method", "heuristic_only"))
    fallback = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    max_steps = int(baseline_cfg.get("max_steps", 0))
    scale_factor = float(baseline_cfg.get("iterations_scale_factor", 1.0))
    if max_steps <= 0:
        max_steps = max(1, int(scale_factor * env.current_solution.S))

    sa_T0 = float(baseline_cfg.get("sa_T0", 1.0))
    sa_alpha = float(baseline_cfg.get("sa_alpha", 0.995))
    selection_frequency = int(baseline_cfg.get("selection_frequency", 5))
    num_candidate_heuristics = int(baseline_cfg.get("num_candidate_heuristics", 4))
    timeout_sec = int(baseline_cfg.get("llm_timeout_s", 30))
    max_retry = int(baseline_cfg.get("max_llm_failures", 1))

    start_time = time.time()
    llm_failed = False

    if method == "llm_hh":
        try:
            hh = LLMSelectionHyperHeuristic(
                env=env,
                heuristics=heuristics,
                rng=rng,
                selection_frequency=selection_frequency,
                num_candidate_heuristics=num_candidate_heuristics,
                sa_T0=sa_T0,
                sa_alpha=sa_alpha,
                timeout_sec=timeout_sec,
                max_retry=max_retry,
            )
            hh.run(max_steps)
            for usage in hh.usage_records:
                _write_usage_line(llm_usage_path, usage)
        except Exception as exc:  # noqa: BLE001
            llm_failed = True
            _write_usage_line(llm_usage_path, {"ok": False, "reason": repr(exc), "fallback": fallback})
            method = fallback

    if method == "random_hh":
        hh = RandomHyperHeuristic(
            env=env,
            heuristics=heuristics,
            rng=rng,
            selection_frequency=selection_frequency,
            sa_T0=sa_T0,
            sa_alpha=sa_alpha,
        )
        hh.run(max_steps)
    elif method == "heuristic_only" or llm_failed:
        hh = RandomHyperHeuristic(
            env=env,
            heuristics=heuristics,
            rng=rng,
            selection_frequency=1,
            sa_T0=sa_T0,
            sa_alpha=sa_alpha,
        )
        hh.run(max_steps)

    evaluator = _build_evaluator(cfg, layout_input)
    layout_state = _build_state(layout_input, initial_assign.copy())
    eval_out = evaluator.evaluate(layout_state)
    best_eval = dict(eval_out)
    best_assign = initial_assign.copy()
    accepted_steps = 0

    trace_path = out_dir / "trace.csv"
    trace_fields = [
        "iter",
        "stage",
        "op",
        "op_args_json",
        "accepted",
        "total_scalar",
        "comm_norm",
        "therm_norm",
        "pareto_added",
        "duplicate_penalty",
        "boundary_penalty",
        "seed_id",
        "time_ms",
        "signature",
        "d_total",
        "d_comm",
        "d_therm",
    ]

    with trace_path.open("w", encoding="utf-8", newline="") as trace_fp:
        writer = csv.writer(trace_fp)
        writer.writerow(trace_fields)
        current_assign = initial_assign.copy()
        for rec in env.recordings:
            operator = rec["operator"]
            action = _action_from_operator(operator)
            candidate_assign = _apply_action(current_assign, action)
            layout_state.assign = candidate_assign
            eval_new = evaluator.evaluate(layout_state)
            d_total = float(eval_new["total_scalar"] - eval_out["total_scalar"])
            d_comm = float(eval_new["comm_norm"] - eval_out["comm_norm"])
            d_therm = float(eval_new["therm_norm"] - eval_out["therm_norm"])
            accepted = bool(rec.get("accepted", False))
            if accepted:
                current_assign = candidate_assign
                eval_out = eval_new
                accepted_steps += 1
                if eval_out["total_scalar"] < best_eval["total_scalar"]:
                    best_eval = dict(eval_out)
                    best_assign = current_assign.copy()
            signature = _assign_signature(current_assign)
            writer.writerow(
                [
                    int(rec.get("step", 0)),
                    "heuragenix",
                    action.get("op", "none"),
                    json.dumps(action),
                    int(accepted),
                    eval_out["total_scalar"],
                    eval_out["comm_norm"],
                    eval_out["therm_norm"],
                    0,
                    eval_out["penalty"]["duplicate"],
                    eval_out["penalty"]["boundary"],
                    int(args.seed),
                    int(rec.get("time_ms", 0)),
                    signature,
                    d_total,
                    d_comm,
                    d_therm,
                ]
            )

    detailed_cfg = cfg.get("detailed_place", {})
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = _compute_trace_metrics(trace_path, metrics_window, eps_flat)

    layout_best = {
        "assign": best_assign.tolist(),
        "best_total": float(best_eval["total_scalar"]),
        "best_comm": float(best_eval["comm_norm"]),
        "best_therm": float(best_eval["therm_norm"]),
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)

    report = {
        "best_total": float(best_eval["total_scalar"]),
        "best_comm": float(best_eval["comm_norm"]),
        "best_therm": float(best_eval["therm_norm"]),
        "steps_total": int(len(env.recordings)),
        "accepts_total": int(accepted_steps),
        "method": method,
        "runtime_s": float(time.time() - start_time),
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        **trace_metrics,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
