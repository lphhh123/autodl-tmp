"""Alternate optimization between mapping and layout (SPEC v4.3.2 §8.7)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from layout.detailed_place import run_detailed_place
from layout.pareto import ParetoSet
from layout.evaluator import LayoutEvaluator, LayoutState
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment


def run_alt_opt(
    rounds: int,
    mapping_solver: MappingSolver,
    segments: List[Segment],
    eff_specs: Dict,
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    assign_init: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    pareto: ParetoSet,
    cfg: Dict,
    trace_path,
    chip_tdp: np.ndarray | None = None,
):
    assign = assign_init.copy()
    mapping = list(range(assign.shape[0]))
    for r in range(rounds):
        # Step1: optional remap (skip if segments missing)
        if segments and eff_specs:
            mapping = mapping_solver.solve_mapping(segments, eff_specs, cfg.get("hw_proxy"), layout_positions=None)["mapping"]
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
    return assign, mapping
