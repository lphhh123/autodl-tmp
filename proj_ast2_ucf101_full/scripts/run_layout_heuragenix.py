"""HeurAgenix baseline wrapper for wafer layout (SPEC v4.3.2/4)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from layout.candidate_pool import _signature_for_action
from layout.evaluator import LayoutEvaluator, LayoutState
from scripts.run_layout_agent import _compute_trace_metrics
from utils.config import load_config


def _load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _init_assign(layout_input: dict, Ns: int, S: int, rng: random.Random) -> np.ndarray:
    if "seed" in layout_input and "assign_seed" in layout_input["seed"]:
        return np.asarray(layout_input["seed"]["assign_seed"], dtype=int)
    if "baseline" in layout_input and "assign_grid" in layout_input["baseline"]:
        return np.asarray(layout_input["baseline"]["assign_grid"], dtype=int)
    slots = int(S)
    sites = np.arange(int(Ns), dtype=int)
    rng.shuffle(sites)
    return sites[:slots]


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


def _random_swap(assign: np.ndarray, rng: random.Random) -> Dict[str, Any]:
    i, j = rng.sample(range(len(assign)), 2)
    return {"op": "swap", "i": int(i), "j": int(j)}


def _random_relocate(assign: np.ndarray, Ns: int, rng: random.Random) -> Dict[str, Any]:
    i = rng.randrange(len(assign))
    site_id = rng.randrange(Ns)
    occupied = {int(s): idx for idx, s in enumerate(assign)}
    if site_id in occupied:
        j = int(occupied[site_id])
        return {"op": "swap", "i": int(i), "j": int(j)}
    return {"op": "relocate", "i": int(i), "site_id": int(site_id), "from_site": int(assign[i])}


def _cluster_move(assign: np.ndarray, Ns: int, rng: random.Random) -> Dict[str, Any]:
    slots = rng.sample(range(len(assign)), min(3, len(assign)))
    target_sites = rng.sample(range(Ns), len(slots))
    return {"op": "cluster_move", "cluster_id": 0, "region_id": 0, "slots": slots, "target_sites": target_sites}


def _evaluate_action(
    evaluator: LayoutEvaluator,
    state: LayoutState,
    assign: np.ndarray,
    action: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    new_assign = _apply_action(assign, action)
    state.assign = new_assign
    eval_new = evaluator.evaluate(state)
    state.assign = assign
    return new_assign, {
        "total": float(eval_new["total_scalar"]),
        "comm": float(eval_new["comm_norm"]),
        "therm": float(eval_new["therm_norm"]),
        "dup_pen": float(eval_new["penalty"]["duplicate"]),
        "bnd_pen": float(eval_new["penalty"]["boundary"]),
    }


def _best_of_candidates(
    evaluator: LayoutEvaluator,
    state: LayoutState,
    assign: np.ndarray,
    candidates: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray]:
    best_action = candidates[0]
    best_eval = None
    best_assign = assign
    best_total = float("inf")
    for cand in candidates:
        cand_assign, cand_eval = _evaluate_action(evaluator, state, assign, cand)
        if cand_eval["total"] < best_total:
            best_total = cand_eval["total"]
            best_action = cand
            best_eval = cand_eval
            best_assign = cand_assign
    return best_action, best_eval, best_assign


def _select_action(
    method: str,
    evaluator: LayoutEvaluator,
    state: LayoutState,
    assign: np.ndarray,
    Ns: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray]:
    if method == "heuristic_only":
        candidates = [_random_swap(assign, rng), _random_relocate(assign, Ns, rng), _cluster_move(assign, Ns, rng)]
        return _best_of_candidates(evaluator, state, assign, candidates)
    if method == "random_hh":
        action = rng.choice([_random_swap(assign, rng), _random_relocate(assign, Ns, rng), _cluster_move(assign, Ns, rng)])
        new_assign, new_eval = _evaluate_action(evaluator, state, assign, action)
        return action, new_eval, new_assign
    action = rng.choice([_random_swap(assign, rng), _random_relocate(assign, Ns, rng), _cluster_move(assign, Ns, rng)])
    new_assign, new_eval = _evaluate_action(evaluator, state, assign, action)
    return action, new_eval, new_assign


def _append_llm_usage(path: Path, payload: Dict[str, Any]) -> None:
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

    seed = int(args.seed)
    _seed_everything(seed)
    rng = random.Random(seed)

    wafer_radius = float(layout_input["wafer"]["radius_mm"])
    sites_xy = np.array(layout_input["sites"]["sites_xy"], dtype=np.float32)
    chip_tdp = np.array(layout_input["slots"]["tdp"], dtype=float)
    traffic = np.array(layout_input["mapping"]["traffic_matrix"], dtype=float)
    S = int(layout_input["slots"].get("S", len(chip_tdp)))
    Ns = int(layout_input["sites"].get("Ns", len(sites_xy)))
    assign = _init_assign(layout_input, Ns, S, rng)

    evaluator = LayoutEvaluator(
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
    layout_state = LayoutState(
        S=S,
        Ns=Ns,
        wafer_radius_mm=wafer_radius,
        sites_xy_mm=sites_xy,
        assign=assign.copy(),
        chip_tdp_w=chip_tdp,
        traffic_bytes=traffic,
        meta={"stage": "seed"},
    )

    eval_out = evaluator.evaluate(layout_state)
    best_eval = dict(eval_out)
    best_assign = assign.copy()

    baseline_cfg = cfg.baseline if hasattr(cfg, "baseline") else {}
    method = str(baseline_cfg.get("method", "heuristic_only"))
    fallback = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    max_steps = int(baseline_cfg.get("max_steps", 0))
    scale_factor = float(baseline_cfg.get("iterations_scale_factor", 1.0))
    if max_steps <= 0:
        max_steps = max(1, int(scale_factor * layout_state.S))

    if method == "llm_hh":
        _append_llm_usage(
            llm_usage_path,
            {
                "ok": False,
                "reason": "llm_unavailable_or_not_configured",
                "fallback": fallback,
                "method": method,
            },
        )
        method = fallback

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

    start_time = time.time()
    accepted_steps = 0
    temperature = float(baseline_cfg.get("sa_T0", 1.0))
    alpha = float(baseline_cfg.get("sa_alpha", 0.995))

    with trace_path.open("w", encoding="utf-8", newline="") as trace_fp:
        writer = csv.writer(trace_fp)
        writer.writerow(trace_fields)
        for step in range(max_steps):
            action, new_eval, new_assign = _select_action(method, evaluator, layout_state, assign, layout_state.Ns, rng)
            d_total = new_eval["total"] - float(eval_out["total_scalar"])
            d_comm = new_eval["comm"] - float(eval_out["comm_norm"])
            d_therm = new_eval["therm"] - float(eval_out["therm_norm"])
            accept = (d_total < 0) or (rng.random() < float(np.exp(-d_total / max(temperature, 1e-6))))
            if accept:
                assign = new_assign
                eval_out = {
                    "total_scalar": new_eval["total"],
                    "comm_norm": new_eval["comm"],
                    "therm_norm": new_eval["therm"],
                    "penalty": {"duplicate": new_eval["dup_pen"], "boundary": new_eval["bnd_pen"]},
                }
                accepted_steps += 1
                if eval_out["total_scalar"] < best_eval["total_scalar"]:
                    best_eval = dict(eval_out)
                    best_assign = assign.copy()

            temperature *= alpha

            action_payload = dict(action)
            action_payload.setdefault("signature", _signature_for_action(action_payload, assign))
            signature = str(action_payload["signature"])
            writer.writerow(
                [
                    step,
                    "heuragenix",
                    action.get("op", "none"),
                    json.dumps(action_payload),
                    int(accept),
                    eval_out["total_scalar"],
                    eval_out["comm_norm"],
                    eval_out["therm_norm"],
                    0,
                    eval_out["penalty"]["duplicate"],
                    eval_out["penalty"]["boundary"],
                    seed,
                    0,
                    signature,
                    d_total,
                    d_comm,
                    d_therm,
                ]
            )
            if step % 20 == 0:
                trace_fp.flush()

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
        "steps_total": int(max_steps),
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
