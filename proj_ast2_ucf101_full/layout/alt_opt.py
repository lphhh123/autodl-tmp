"""Alternate optimization between mapping and layout (SPEC v5.4 §8.7)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from layout.detailed_place import run_detailed_place
from layout.pareto import ParetoSet
from layout.evaluator import LayoutEvaluator, LayoutState
from mapping.segments import Segment


def _top_segments_by_traffic(segments: List[Segment], ratio: float) -> List[int]:
    if not segments:
        return []
    ratio = max(0.0, min(1.0, ratio))
    k = max(1, int(len(segments) * ratio))
    order = sorted(range(len(segments)), key=lambda i: getattr(segments[i], "traffic_out_bytes", 0.0), reverse=True)
    return order[:k]


def _remap_top_segments(mapping: List[int], segments: List[Segment], assign: np.ndarray, sites_xy: np.ndarray, ratio: float) -> List[int]:
    if not segments or not mapping:
        return mapping
    S = assign.shape[0]
    mapping = [m % S for m in mapping]
    top_ids = _top_segments_by_traffic(segments, ratio)
    pos = sites_xy[assign]

    def _edge_cost(idx_a: int, idx_b: int, map_a: int, map_b: int) -> float:
        if idx_a < 0 or idx_b >= len(mapping):
            return 0.0
        traffic = getattr(segments[idx_a], "traffic_out_bytes", 0.0)
        if map_a == map_b:
            return 0.0
        return traffic * float(np.linalg.norm(pos[map_a] - pos[map_b]))

    for i in top_ids:
        curr_slot = mapping[i]
        best_slot = curr_slot
        best_delta = 0.0
        for s in range(S):
            if s == curr_slot:
                continue
            delta = 0.0
            delta += _edge_cost(i - 1, i, mapping[i - 1] if i - 1 >= 0 else s, s)
            delta += _edge_cost(i, i + 1, s, mapping[i + 1] if i + 1 < len(mapping) else s)
            delta -= _edge_cost(i - 1, i, mapping[i - 1] if i - 1 >= 0 else curr_slot, curr_slot)
            delta -= _edge_cost(i, i + 1, curr_slot, mapping[i + 1] if i + 1 < len(mapping) else curr_slot)
            if delta < best_delta:
                best_delta = delta
                best_slot = s
        mapping[i] = best_slot
    return mapping


def _comm_cost(mapping: List[int], segments: List[Segment], assign: np.ndarray, sites_xy: np.ndarray) -> float:
    """Compute simple comm-only cost using consecutive segment traffic and site distance."""

    if not mapping or len(mapping) <= 1:
        return 0.0
    pos = sites_xy[assign]
    cost = 0.0
    for k in range(len(mapping) - 1):
        a, b = mapping[k], mapping[k + 1]
        if a == b:
            continue
        traffic = getattr(segments[k], "traffic_out_bytes", 0.0) if k < len(segments) else 0.0
        cost += traffic * float(np.linalg.norm(pos[a] - pos[b]))
    return cost


def _comm_only_swap(
    mapping: List[int],
    segments: List[Segment],
    assign: np.ndarray,
    sites_xy: np.ndarray,
    max_swaps: int = 32,
) -> Tuple[List[int], Dict]:
    """Greedy swap to reduce comm distance without hardware proxy."""

    mapping = list(mapping)
    if not mapping or len(mapping) <= 1:
        return mapping, {"swaps": 0, "mode": "noop"}
    S = assign.shape[0]
    mapping = [m % S for m in mapping]
    best_cost = _comm_cost(mapping, segments, assign, sites_xy)
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        for i in range(len(mapping)):
            for j in range(i + 1, len(mapping)):
                mapping[i], mapping[j] = mapping[j], mapping[i]
                cost = _comm_cost(mapping, segments, assign, sites_xy)
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    swaps += 1
                    improved = True
                else:
                    mapping[i], mapping[j] = mapping[j], mapping[i]
            if swaps >= max_swaps:
                break
    return mapping, {"swaps": swaps, "mode": "comm_only_swap", "cost": best_cost}


def run_alt_opt(
    rounds: int,
    segments: List[Segment],
    mapping_init: Optional[List[int]],
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    assign_init: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    pareto: ParetoSet,
    cfg: Dict,
    trace_path,
    chip_tdp: np.ndarray | None = None,
    mapping_solver=None,
):
    assign = assign_init.copy()
    mapping = list(mapping_init) if mapping_init else list(range(assign.shape[0]))
    mapping_mode = cfg.get("mapping_mode", "comm_only") if cfg else "comm_only"
    mapping_step_meta: Dict = {"mapping_step": "skipped_missing_solver_or_segments", "mapping_mode": mapping_mode}
    for r in range(rounds):
        # Step1: optional remap (skip if segments missing)
        if mapping_solver is not None and segments and mapping_mode != "skip":
            mapping, mapping_step_meta = _comm_only_swap(
                mapping=mapping,
                segments=segments,
                assign=assign,
                sites_xy=sites_xy,
                max_swaps=int(cfg.get("max_mapping_swaps", 32)),
            )
            mapping_step_meta["mapping_mode"] = mapping_mode
            mapping_step_meta["round"] = r
        else:
            mapping_step_meta = {
                "mapping_step": "skipped_missing_solver_or_segments",
                "mapping_mode": mapping_mode,
                "round": r,
            }
        layout_state.assign = assign
        # Step2: refine layout starting from Pareto seeds
        seeds = pareto.points[: max(1, min(5, len(pareto.points)))]
        seeds_assign = [assign] if not seeds else [p.payload.get("assign", assign) for p in seeds]
        for idx, seed_assign in enumerate(seeds_assign):
            layout_state.assign = seed_assign
            result = run_detailed_place(
                sites_xy=sites_xy,
                assign_seed=seed_assign,
                evaluator=evaluator,
                layout_state=layout_state,
                traffic_sym=traffic_sym,
                site_to_region=np.zeros((sites_xy.shape[0],), dtype=int),
                regions=[],
                clusters=[],
                cluster_to_region=[],
                pareto=pareto,
                cfg=cfg.get("refine_each_round", {}),
                trace_path=trace_path.parent / f"alt_opt_round_{r}_{idx}.csv",
                seed_id=r,
                chip_tdp=chip_tdp,
            )
            assign = result.assign
    if pareto.points:
        for p in pareto.points:
            payload = p.payload
            payload.setdefault("alt_opt", {})
            payload["alt_opt"].update(mapping_step_meta)
    return assign, mapping
