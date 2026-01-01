"""Detailed placement with SA and Pareto updates."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Tuple

import numpy as np

from .evaluator import LayoutEvaluator, LayoutState
from .llm_provider import HeuristicProvider, LLMProvider, VolcArkProvider
from .pareto import ParetoSet


def _sample_action(S: int, action_probs: Dict[str, float], rng: np.random.Generator) -> str:
    ops = list(action_probs.keys())
    probs = np.array([action_probs[k] for k in ops], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(ops, p=probs))


def _assign_hash(assign: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(assign, dtype=np.int64).tobytes()).hexdigest()[:16]


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
    enable_pareto: bool = True,
    planner_cfg: Dict | None = None,
    llm_provider: LLMProvider | None = None,
    llm_usage: List[Dict] | None = None,
    rng: np.random.Generator | None = None,
    trace: List[Dict] | None = None,
    pareto_points: List[Dict] | None = None,
    seed_id: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """SA loop with swap/relocate moves, Pareto updates and optional LLM planner."""

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

    planner_cfg = planner_cfg or {}
    planner_type = str(planner_cfg.get("type", "heuristic"))
    mixed_every_n = int(planner_cfg.get("mixed", {}).get("every_n_steps", 50))
    k_actions = int(planner_cfg.get("mixed", {}).get("k_actions", 4))
    if planner_type in {"llm", "mixed"} and llm_provider is None:
        llm_provider = VolcArkProvider(
            timeout_sec=int(planner_cfg.get("timeout_sec", 30)),
            max_retry=int(planner_cfg.get("max_retry", 2)),
        )
    if llm_provider is None:
        llm_provider = HeuristicProvider()

    traffic_sym = traffic_bytes + traffic_bytes.T

    def build_state_summary(eval_res: Dict) -> Dict:
        # build hot pairs with current distances
        dists = {}
        pos = sites_xy_mm[cur_assign]
        for i, j in hot_pairs:
            dists[(i, j)] = float(np.linalg.norm(pos[i] - pos[j]))
        top_pairs = [
            {"i": i, "j": j, "traffic": float(traffic_sym[i, j]), "dist_mm": dists.get((i, j), 0.0)}
            for i, j in hot_pairs
        ]
        top_slots = [
            {"i": int(i), "tdp": float(t), "region": None}
            for i, t in sorted(enumerate(chip_tdp_w), key=lambda x: x[1], reverse=True)[: max(1, len(hot_pairs))]
        ]
        return {
            "comm_norm": float(eval_res.get("comm_norm", 0.0)),
            "therm_norm": float(eval_res.get("therm_norm", 0.0)),
            "top_hot_pairs": top_pairs,
            "top_hot_slots": top_slots,
            "violations": eval_res.get("penalty", {}),
            "hint": "prefer swap on hot pairs; push high-tdp outward; avoid duplicate sites",
        }

    def apply_action(base: np.ndarray, action: Dict, rng_local: np.random.Generator) -> np.ndarray:
        cand = base.copy()
        op = action.get("op")
        if op == "swap":
            i, j = int(action.get("i", -1)), int(action.get("j", -1))
            if 0 <= i < S and 0 <= j < S:
                cand[i], cand[j] = cand[j], cand[i]
        elif op == "relocate":
            i = int(action.get("i", -1))
            site_id = action.get("site_id")
            if 0 <= i < S:
                occupied = set(int(x) for x in cand.tolist())
                if site_id is None or site_id in occupied or not (0 <= int(site_id) < Ns):
                    free = [s for s in range(Ns) if s not in occupied]
                    if free:
                        site_id = int(rng_local.choice(free))
                if site_id is not None and 0 <= int(site_id) < Ns:
                    cand[i] = int(site_id)
        elif op == "cluster_move":
            slots = action.get("slots")
            if not slots:
                slots = list(rng_local.choice(S, size=max(1, S // 4), replace=False))
            target_sites = action.get("site_ids")
            occupied = set(int(x) for x in cand.tolist())
            free = [s for s in range(Ns) if s not in occupied]
            if target_sites:
                target_sites = [int(s) for s in target_sites if 0 <= int(s) < Ns and int(s) not in occupied]
            else:
                target_sites = free[: len(slots)]
            if len(target_sites) >= len(slots):
                for sl, site in zip(slots, target_sites):
                    cand[int(sl)] = int(site)
        return cand

    cur_eval = eval_assign(cur_assign)
    best_assign = cur_assign.copy()
    best_eval = cur_eval

    for step in range(steps):
        use_llm = planner_type in {"llm", "mixed"} and (planner_type == "llm" or step % max(1, mixed_every_n) == 0)
        cand = cur_assign.copy()
        op = "none"
        if use_llm and llm_provider is not None:
            state_summary = build_state_summary(cur_eval)
            actions, usage = llm_provider.propose_actions(state_summary, k_actions)
            if usage and llm_usage is not None:
                usage = dict(usage)
                usage.update({"step": step})
                llm_usage.append(usage)
            for act in actions:
                cand = apply_action(cur_assign, act, rng)
                op = act.get("op", "llm")
                break
        if op == "none":
            op = _sample_action(S, action_probs, rng)
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

        added = False
        if enable_pareto:
            added = pareto.try_add(
                comm_norm=cand_eval["comm_norm"],
                therm_norm=cand_eval["therm_norm"],
                total_scalar=cand_eval["total_scalar"],
                meta={
                    "stage": planner_cfg.get("stage_label", "detailed"),
                    "iter": step,
                    "seed_id": seed_id,
                    "assign_hash": _assign_hash(cand),
                },
            )
            if added and pareto_points is not None:
                pareto_points.append(
                    {
                        "solution_id": len(pareto_points),
                        "comm_norm": cand_eval["comm_norm"],
                        "therm_norm": cand_eval["therm_norm"],
                        "total_scalar": cand_eval["total_scalar"],
                        "stage": planner_cfg.get("stage_label", "detailed"),
                        "iter": step,
                        "seed_id": seed_id,
                        "assign_hash": _assign_hash(cand),
                    }
                )
        trace.append(
            {
                "iter": step,
                "stage": planner_cfg.get("stage_label", "detailed"),
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

