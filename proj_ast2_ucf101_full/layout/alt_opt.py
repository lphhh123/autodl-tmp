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
    sites_xy: np.ndarray,
    assign_init: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    pareto: ParetoSet,
    cfg: Dict,
    trace_path,
):
    assign = assign_init.copy()
    mapping = list(range(len(segments)))
    for r in range(rounds):
        # Step1: remap limited segments (placeholder greedy)
        mapping = mapping_solver.solve_mapping(segments, eff_specs, cfg.get("hw_proxy"), layout_positions=None)["mapping"]
        layout_state.assign = assign
        # Step2: refine layout starting from top pareto points
        result = run_detailed_place(
            sites_xy=sites_xy,
            assign_seed=assign,
            evaluator=evaluator,
            layout_state=layout_state,
            traffic_sym=np.zeros((layout_state.S, layout_state.S)),
            site_to_region=np.zeros((sites_xy.shape[0],), dtype=int),
            regions=[],
            clusters=[],
            cluster_to_region=[],
            pareto=pareto,
            cfg=cfg.get("refine_each_round", {}),
            trace_path=trace_path.parent / f"alt_opt_round_{r}.csv",
            seed_id=r,
        )
        assign = result.assign
    return assign, mapping
