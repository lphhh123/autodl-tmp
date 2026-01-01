"""Alternate optimization between mapping and layout (SPEC v4.3.2 §8.7)."""
from __future__ import annotations

from typing import Dict, List

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


def run_alt_opt(
    rounds: int,
    segments: List[Segment],
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    assign_init: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    pareto: ParetoSet,
    cfg: Dict,
    trace_dir,
    chip_tdp: np.ndarray | None = None,
):
    assign = assign_init.copy()
    mapping = list(range(assign.shape[0])) if not cfg.get("mapping", None) else cfg.get("mapping")
    allow_ratio = float(cfg.get("remap", {}).get("allow_top_segment_ratio", 0.2))
    refine_cfg = cfg.get("refine_each_round", {})
    for r in range(rounds):
        # Step1: remap a limited portion of bottleneck segments based on current layout distances
        mapping = _remap_top_segments(mapping, segments, assign, sites_xy, allow_ratio)
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
                cfg=refine_cfg,
                trace_path=trace_dir / f"alt_opt_round_{r}_{idx}.csv",
                seed_id=r,
                chip_tdp=chip_tdp,
                stage_label=f"alt_opt_round_{r}",
            )
            assign = result.assign
    return assign, mapping
