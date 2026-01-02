"""Detailed placement with SA and Pareto updates (SPEC v4.3.2 §8.6)."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _sample_action(
    cfg: Dict,
    traffic_sym: np.ndarray,
    site_to_region: np.ndarray,
    regions,
    clusters: List[Cluster],
    assign: np.ndarray,
    sites_xy: np.ndarray,
    chip_tdp: Optional[np.ndarray],
    cluster_to_region: List[int],
):
    probs = cfg.get("action_probs", {})
    # swap: prioritize hot pairs
    if random.random() < probs.get("swap", 0.5):
        top_pairs = _compute_top_pairs(traffic_sym, int(cfg.get("hot_sampling", {}).get("top_pairs_k", 10)))
        if top_pairs:
            i, j, _ = random.choice(top_pairs)
            return {"op": "swap", "i": int(i), "j": int(j)}
    # relocate: prefer empty sites within the same region
    if random.random() < probs.get("relocate", 0.3):
        same_region_prob = cfg.get("relocate", {}).get("same_region_prob", 0.8)
        neighbor_k = int(cfg.get("relocate", {}).get("neighbor_k", 30))
        # pick a high-traffic or high-tdp slot to move
        slot_scores = traffic_sym.sum(axis=1)
        if chip_tdp is not None and len(chip_tdp) == slot_scores.shape[0]:
            slot_scores = slot_scores + chip_tdp
        slot = int(np.argmax(slot_scores)) if slot_scores.sum() > 0 else random.randrange(traffic_sym.shape[0])
        current_region = site_to_region[assign[slot]]
        empty_sites = [s for s in range(site_to_region.shape[0]) if s not in assign]
        if not empty_sites:
            return {"op": "none"}
        candidates = [s for s in empty_sites if site_to_region[s] == current_region]
        if not candidates or random.random() > same_region_prob:
            candidates = empty_sites
        # choose nearest candidate sites
        dists = [(sid, np.linalg.norm(sites_xy[assign[slot]] - sites_xy[sid])) for sid in candidates]
        dists.sort(key=lambda x: x[1])
        chosen = dists[: max(1, min(neighbor_k, len(dists)))]
        site_id = random.choice(chosen)[0]
        return {"op": "relocate", "i": slot, "site_id": int(site_id)}
    # cluster_move: move heavy cluster outward when capacity allows
    if clusters:
        clusters_sorted = sorted(clusters, key=lambda c: c.tdp_sum, reverse=True)
        c = clusters_sorted[0]
        target_region = regions[-1] if regions else None
        if target_region is not None and cluster_to_region:
            # try a different region than current assignment
            cur_region = cluster_to_region[c.cluster_id] if c.cluster_id < len(cluster_to_region) else -1
            region_options = [r for r in regions if r.region_id != cur_region]
            if region_options:
                target_region = random.choice(region_options)
        if target_region is not None:
            return {"op": "cluster_move", "cluster_id": c.cluster_id, "region_id": target_region.region_id}
    return {"op": "none"}


def _state_summary(
    comm_norm: float,
    therm_norm: float,
    traffic_sym: np.ndarray,
    assign: np.ndarray,
    site_to_region: np.ndarray,
    chip_tdp: Optional[np.ndarray],
) -> Dict:
    pairs = _compute_top_pairs(traffic_sym, 5)
    hot_slots = []
    if chip_tdp is not None and len(chip_tdp) == traffic_sym.shape[0]:
        order = np.argsort(chip_tdp)[::-1][:5]
        hot_slots = [{"i": int(i), "tdp": float(chip_tdp[i]), "region": int(site_to_region[assign[i]])} for i in order]
    return {
        "comm_norm": comm_norm,
        "therm_norm": therm_norm,
        "top_hot_pairs": [{"i": i, "j": j, "traffic": t} for i, j, t in pairs],
        "top_hot_slots": hot_slots,
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
    chip_tdp: Optional[np.ndarray] = None,
    llm_usage_path: Optional[Path] = None,
):
    base_seed = int(cfg.get("seed", 0)) + int(seed_id)
    rng = np.random.default_rng(base_seed)
    random.seed(base_seed)
    assign = assign_seed.copy()
    layout_state.assign = assign
    assert layout_state.assign.shape[0] == layout_state.S
    assert np.all(layout_state.assign >= 0)
    planner_cfg = cfg.get("planner", {"type": "heuristic"})
    planner_type = str(planner_cfg.get("type", "heuristic")).lower()

    heur_provider = HeuristicProvider()
    llm_provider: Optional[LLMProvider] = None
    if planner_type in ("llm", "mixed"):
        llm_provider = VolcArkProvider(
            timeout_sec=int(planner_cfg.get("timeout_sec", 30)),
            max_retry=int(planner_cfg.get("max_retry", 2)),
        )

    mixed_cfg = planner_cfg.get("mixed", {}) if planner_type == "mixed" else {}
    mixed_every = int(mixed_cfg.get("every_n_steps", 50)) if planner_type == "mixed" else 0
    k_actions = int(mixed_cfg.get("k_actions", planner_cfg.get("k_actions", 4)))
    stage_label = str(cfg.get("stage_label", f"detailed_{planner_type}"))

    steps = int(cfg.get("steps", 0))
    T = float(cfg.get("sa_T0", 1.0))
    alpha = float(cfg.get("sa_alpha", 0.999))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    usage_fp = llm_usage_path.open("a", encoding="utf-8") if llm_usage_path else None
    with trace_path.open("w", encoding="utf-8") as f_trace:
        f_trace.write(
            "iter,stage,op,op_args_json,accepted,total_scalar,comm_norm,therm_norm,pareto_added,duplicate_penalty,boundary_penalty,seed_id,time_ms\n"
        )
        eval_out = evaluator.evaluate(layout_state)
        pareto.add(
            eval_out["comm_norm"],
            eval_out["therm_norm"],
            {
                "assign": assign.copy(),
                "total_scalar": eval_out["total_scalar"],
                "stage": stage_label,
                "iter": 0,
                "seed": seed_id,
            },
        )
        for step in range(steps):
            # choose action
            actions: List[Dict] = []
            usage_entry: Optional[Dict] = None
            attempt_llm = planner_type == "llm" or (planner_type == "mixed" and mixed_every > 0 and step % mixed_every == 0)
            ss: Optional[Dict] = None
            if attempt_llm:
                ss = _state_summary(
                    eval_out["comm_norm"], eval_out["therm_norm"], traffic_sym, assign, site_to_region, chip_tdp
                )
                if llm_provider is None:
                    usage_entry = {
                        "step": step,
                        "mode": planner_type,
                        "ok": False,
                        "skipped": True,
                        "reason": "llm_provider_not_initialized",
                    }
                else:
                    try:
                        actions = llm_provider.propose_actions(ss, k_actions)
                        usage_entry = {"step": step, "mode": planner_type}
                        usage_info = getattr(llm_provider, "last_usage", None) or {}
                        usage_entry.update(usage_info)
                        usage_entry.setdefault("ok", bool(actions))
                    except Exception as exc:  # noqa: BLE001
                        usage_entry = {
                            "step": step,
                            "mode": planner_type,
                            "ok": False,
                            "error": str(exc),
                        }
                        if hasattr(llm_provider, "last_usage") and llm_provider.last_usage:
                            usage_entry["usage"] = llm_provider.last_usage
                if usage_fp:
                    json.dump(usage_entry, usage_fp)
                    usage_fp.write("\n")

            if not actions:
                if planner_type in ("heuristic", "mixed"):
                    if ss is None:
                        ss = _state_summary(
                            eval_out["comm_norm"], eval_out["therm_norm"], traffic_sym, assign, site_to_region, chip_tdp
                        )
                    actions = heur_provider.propose_actions(ss, k_actions)
                action = actions[0] if actions else _sample_action(
                    cfg, traffic_sym, site_to_region, regions, clusters, assign, sites_xy, chip_tdp, cluster_to_region
                )
            else:
                action = actions[0]

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
                added = pareto.add(
                    eval_out["comm_norm"],
                    eval_out["therm_norm"],
                    {
                        "assign": assign.copy(),
                        "total_scalar": eval_out["total_scalar"],
                        "stage": stage_label,
                        "iter": step + 1,
                        "seed": seed_id,
                    },
                )
            else:
                layout_state.assign = assign
                added = False
            T *= alpha
            f_trace.write(
                f"{step},{stage_label},{op},{json.dumps(action)},"
                f"{int(accept)},{eval_out['total_scalar']},{eval_out['comm_norm']},{eval_out['therm_norm']},{int(added)},"
                f"{eval_out['penalty']['duplicate']},{eval_out['penalty']['boundary']},{seed_id},0\n"
            )
    if usage_fp:
        usage_fp.close()
    return DetailedPlaceResult(assign=assign, pareto=pareto, trace_path=trace_path)
