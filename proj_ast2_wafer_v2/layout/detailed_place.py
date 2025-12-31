"""Detailed placement with SA and Pareto updates."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import numpy as np

from .evaluator import LayoutEvaluator, LayoutState
from .pareto import ParetoSet


def _sample_action(S: int, action_probs: Dict[str, float], rng: np.random.Generator) -> str:
    ops = list(action_probs.keys())
    probs = np.array([action_probs[k] for k in ops], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(ops, p=probs))


def run_detailed_place(
    assign: np.ndarray,
    evaluator: LayoutEvaluator,
    sites_xy_mm: np.ndarray,
    chip_tdp_w: np.ndarray,
    traffic_bytes: np.ndarray,
    pareto: ParetoSet,
    steps: int,
    sa_T0: float,
    sa_alpha: float,
    action_probs: Dict[str, float],
    hot_pairs: List[Tuple[int, int]],
    rng: np.random.Generator | None = None,
    trace: List[Dict] | None = None,
    seed_id: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """SA loop with swap/relocate moves and Pareto updates."""

    if rng is None:
        rng = np.random.default_rng()
    if trace is None:
        trace = []

    S = assign.shape[0]
    Ns = sites_xy_mm.shape[0]
    cur_assign = assign.copy()

    def eval_assign(a: np.ndarray) -> Dict:
        return evaluator.evaluate(
            LayoutState(
                S=S,
                Ns=Ns,
                wafer_radius_mm=0.0,  # boundary handled upstream
                sites_xy_mm=sites_xy_mm,
                assign=a,
                chip_tdp_w=chip_tdp_w,
                traffic_bytes=traffic_bytes,
                meta={},
            )
        )

    cur_eval = eval_assign(cur_assign)
    best_assign = cur_assign.copy()
    best_eval = cur_eval

    for step in range(steps):
        op = _sample_action(S, action_probs, rng)
        cand = cur_assign.copy()
        if op == "swap":
            if hot_pairs:
                i, j = hot_pairs[step % len(hot_pairs)][:2]
            else:
                i, j = rng.choice(S, size=2, replace=False)
            cand[i], cand[j] = cand[j], cand[i]
        elif op == "relocate":
            i = int(rng.integers(0, S))
            occupied = set(int(x) for x in cand.tolist())
            free = [s for s in range(Ns) if s not in occupied]
            if free:
                cand[i] = int(rng.choice(free))
        elif op == "cluster_move":
            # lightweight proxy: random subset relocate to random free sites
            size = max(1, S // 4)
            slots = rng.choice(S, size=size, replace=False)
            occupied = set(int(x) for x in cand.tolist())
            free = [s for s in range(Ns) if s not in occupied]
            if len(free) >= len(slots):
                for sl, site in zip(slots, free[: len(slots)]):
                    cand[sl] = int(site)

        cand_eval = eval_assign(cand)
        delta = cand_eval["total_scalar"] - cur_eval["total_scalar"]
        T = sa_T0 * (sa_alpha ** step)
        accept = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        if accept:
            cur_assign = cand
            cur_eval = cand_eval
            if cand_eval["total_scalar"] < best_eval["total_scalar"]:
                best_assign = cand
                best_eval = cand_eval

        added = pareto.try_add(
            comm_norm=cand_eval["comm_norm"],
            therm_norm=cand_eval["therm_norm"],
            total_scalar=cand_eval["total_scalar"],
            meta={"stage": "detailed", "iter": step, "seed_id": seed_id},
        )
        trace.append(
            {
                "iter": step,
                "stage": "detailed",
                "op": op,
                "op_args_json": json.dumps({}),
                "accepted": int(accept),
                "total_scalar": cand_eval["total_scalar"],
                "comm_norm": cand_eval["comm_norm"],
                "therm_norm": cand_eval["therm_norm"],
                "pareto_added": int(added),
                "duplicate_penalty": cand_eval["penalty"]["duplicate"],
                "boundary_penalty": cand_eval["penalty"]["boundary"],
                "seed_id": seed_id,
                "time_ms": 0,
            }
        )

    return best_assign, best_eval

