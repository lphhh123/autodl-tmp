"""Detailed placement with SA and Pareto updates (SPEC v4.3.2 §8.6)."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from layout.coarsen import Cluster
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from layout.llm_provider import HeuristicProvider, VolcArkProvider, LLMProvider


@dataclass
class DetailedPlaceResult:
    assign: np.ndarray
    pareto: ParetoSet
    trace_path: Path


def _compute_top_pairs(traffic_sym: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    S = traffic_sym.shape[0]
    for i in range(S):
        for j in range(i + 1, S):
            pairs.append((i, j, float(traffic_sym[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def _apply_swap(assign: np.ndarray, i: int, j: int):
    assign[i], assign[j] = assign[j], assign[i]


def _apply_relocate(assign: np.ndarray, i: int, site_id: int):
    assign[i] = site_id


def _apply_cluster_move(assign: np.ndarray, cluster: Cluster, target_sites: List[int]):
    if len(target_sites) < len(cluster.slots):
        return
    for slot, site in zip(cluster.slots, target_sites):
        assign[slot] = site


def _sample_action(cfg: Dict, traffic_sym: np.ndarray, site_to_region: np.ndarray, regions, clusters: List[Cluster], assign: np.ndarray):
    probs = cfg.get("action_probs", {})
    r = random.random()
    if r < probs.get("swap", 0.5):
        top_pairs = _compute_top_pairs(traffic_sym, int(cfg.get("hot_sampling", {}).get("top_pairs_k", 10)))
        if top_pairs:
            i, j, _ = random.choice(top_pairs)
            return {"op": "swap", "i": int(i), "j": int(j)}
    r = random.random()
    if r < probs.get("relocate", 0.3):
        i = random.randrange(traffic_sym.shape[0])
        # prefer same region empty site
        same_region_prob = cfg.get("relocate", {}).get("same_region_prob", 0.8)
        region_id = site_to_region[assign[i]]
        candidate_sites = [idx for idx, rid in enumerate(site_to_region) if rid == region_id]
        if not candidate_sites:
            candidate_sites = list(range(site_to_region.shape[0]))
        site_id = random.choice(candidate_sites)
        return {"op": "relocate", "i": i, "site_id": int(site_id)}
    # cluster_move fallback
    if clusters:
        c = random.choice(clusters)
        target_region = random.choice(regions)
        return {"op": "cluster_move", "cluster_id": c.cluster_id, "region_id": target_region.region_id}
    return {"op": "none"}


def _init_provider(planner_cfg: Dict) -> LLMProvider:
    planner_type = planner_cfg.get("type", "heuristic")
    if planner_type == "llm":
        return VolcArkProvider(
            timeout_sec=int(planner_cfg.get("timeout_sec", 30)),
            max_retry=int(planner_cfg.get("max_retry", 2)),
        )
    if planner_type == "mixed":
        # mixed handled in caller by alternating heuristic/llm
        return HeuristicProvider()
    return HeuristicProvider()


def _state_summary(comm_norm: float, therm_norm: float, traffic_sym: np.ndarray, assign: np.ndarray, site_to_region: np.ndarray) -> Dict:
    pairs = _compute_top_pairs(traffic_sym, 5)
    return {
        "comm_norm": comm_norm,
        "therm_norm": therm_norm,
        "top_hot_pairs": [{"i": i, "j": j, "traffic": t} for i, j, t in pairs],
        "top_hot_slots": [],
        "violations": {"duplicate": 0, "boundary": 0},
        "hint": "prefer swap on hot pairs; push high-tdp outward; avoid duplicate sites",
    }


def run_detailed_place(
    sites_xy: np.ndarray,
    assign_seed: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    traffic_sym: np.ndarray,
    site_to_region: np.ndarray,
    regions,
    clusters: List[Cluster],
    cluster_to_region: List[int],
    pareto: ParetoSet,
    cfg: Dict,
    trace_path: Path,
    seed_id: int,
):
    rng = np.random.default_rng(cfg.get("seed", 0) + seed_id)
    assign = assign_seed.copy()
    planner_cfg = cfg.get("planner", {"type": "heuristic"})
    planner = _init_provider(planner_cfg)
    mixed_every = int(planner_cfg.get("mixed", {}).get("every_n_steps", 50)) if planner_cfg.get("type") == "mixed" else None
    k_actions = int(planner_cfg.get("mixed", {}).get("k_actions", 4))

    steps = int(cfg.get("steps", 0))
    T = float(cfg.get("sa_T0", 1.0))
    alpha = float(cfg.get("sa_alpha", 0.999))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f_trace:
        f_trace.write(
            "iter,stage,op,op_args_json,accepted,total_scalar,comm_norm,therm_norm,pareto_added,duplicate_penalty,boundary_penalty,seed_id,time_ms\n"
        )
        eval_out = evaluator.evaluate(layout_state)
        pareto.add(eval_out["comm_norm"], eval_out["therm_norm"], {"assign": assign.copy()})
        for step in range(steps):
            # choose action
            if planner_cfg.get("type") == "mixed" and mixed_every and step % mixed_every == 0:
                ss = _state_summary(eval_out["comm_norm"], eval_out["therm_norm"], traffic_sym, assign, site_to_region)
                try:
                    actions = planner.propose_actions(ss, k_actions)
                except Exception:
                    actions = []
                action = actions[0] if actions else _sample_action(cfg, traffic_sym, site_to_region, regions, clusters, assign)
            else:
                action = _sample_action(cfg, traffic_sym, site_to_region, regions, clusters, assign)

            new_assign = assign.copy()
            op = action.get("op", "none")
            if op == "swap":
                _apply_swap(new_assign, int(action.get("i", 0)), int(action.get("j", 0)))
            elif op == "relocate":
                _apply_relocate(new_assign, int(action.get("i", 0)), int(action.get("site_id", 0)))
            elif op == "cluster_move":
                cid = int(action.get("cluster_id", 0))
                rid = int(action.get("region_id", 0))
                if cid < len(clusters):
                    cluster = clusters[cid]
                    target_sites = [s for s, r in enumerate(site_to_region) if r == rid][: len(cluster.slots)]
                    _apply_cluster_move(new_assign, cluster, target_sites)

            layout_state.assign = new_assign
            eval_new = evaluator.evaluate(layout_state)
            delta = eval_new["total_scalar"] - eval_out["total_scalar"]
            accept = delta < 0 or math.exp(-delta / max(T, 1e-6)) > rng.random()
            if accept:
                assign = new_assign
                eval_out = eval_new
                layout_state.assign = assign
                added = pareto.add(eval_out["comm_norm"], eval_out["therm_norm"], {"assign": assign.copy()})
            else:
                layout_state.assign = assign
                added = False
            T *= alpha
            f_trace.write(
                f"{step},detailed,{op},{json.dumps(action)}," f"{int(accept)},{eval_out['total_scalar']},{eval_out['comm_norm']},{eval_out['therm_norm']},{int(added)},{eval_out['penalty']['duplicate']},{eval_out['penalty']['boundary']},{seed_id},0\n"
            )
    return DetailedPlaceResult(assign=assign, pareto=pareto, trace_path=trace_path)
