"""Offline EDA-Agent driver matching SPEC v4.3.2."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from layout.alt_opt import run_alt_opt
from layout.coarsen import coarsen_traffic
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.expand import expand_clusters
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.regions import build_regions
from mapping.mapping_solver import MappingSolver
from utils.config import load_config


@dataclass
class TraceLogger:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._f.write(
            "iter,stage,op,op_args_json,accepted,total_scalar,comm_norm,therm_norm,pareto_added,duplicate_penalty,boundary_penalty,seed_id,time_ms\n"
        )

    def log(
        self,
        *,
        iter_idx: int,
        stage: str,
        op: str,
        op_args: Dict,
        accepted: int,
        total_scalar: float,
        comm_norm: float,
        therm_norm: float,
        pareto_added: int,
        penalty_dup: float,
        penalty_bound: float,
        seed_id: int,
        time_ms: float = 0.0,
    ):
        self._f.write(
            f"{iter_idx},{stage},{op},{json.dumps(op_args)},{accepted},{total_scalar},{comm_norm},{therm_norm},{pareto_added},{penalty_dup},{penalty_bound},{seed_id},{time_ms}\n"
        )

    def close(self):
        self._f.close()


def load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_and_log(
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    pareto: ParetoSet,
    pareto_records: List[Dict],
    trace: TraceLogger,
    stage: str,
    seed_id: int,
    iter_idx: int,
):
    eval_out = evaluator.evaluate(layout_state)
    added = pareto.add(
        eval_out["comm_norm"],
        eval_out["therm_norm"],
        {"assign": layout_state.assign.copy(), "total_scalar": eval_out["total_scalar"], "stage": stage, "iter": iter_idx, "seed_id": seed_id},
    )
    if added:
        pareto_records.append(
            {
                "solution_id": len(pareto_records),
                "comm_norm": eval_out["comm_norm"],
                "therm_norm": eval_out["therm_norm"],
                "total_scalar": eval_out["total_scalar"],
                "stage": stage,
                "iter": iter_idx,
                "seed_id": seed_id,
                "assign_hash": hash(tuple(layout_state.assign.tolist())),
            }
        )
    trace.log(
        iter_idx=iter_idx,
        stage=stage,
        op="none",
        op_args={},
        accepted=1,
        total_scalar=eval_out["total_scalar"],
        comm_norm=eval_out["comm_norm"],
        therm_norm=eval_out["therm_norm"],
        pareto_added=int(added),
        penalty_dup=eval_out["penalty"]["duplicate"],
        penalty_bound=eval_out["penalty"]["boundary"],
        seed_id=seed_id,
    )


def _write_pareto_points(pareto_records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("solution_id,comm_norm,therm_norm,total_scalar,stage,iter,seed_id,assign_hash\n")
        for rec in pareto_records:
            f.write(
                f"{rec['solution_id']},{rec['comm_norm']},{rec['therm_norm']},{rec['total_scalar']},{rec['stage']},{rec['iter']},{rec['seed_id']},{rec['assign_hash']}\n"
            )


def _build_evaluator(layout_input: dict, cfg) -> Tuple[LayoutEvaluator, dict]:
    baseline = {
        "L_comm_baseline": float(layout_input["baseline"].get("L_comm", 1.0)),
        "L_therm_baseline": float(layout_input["baseline"].get("L_therm", 1.0)),
    }
    scalar_w = {
        "w_comm": float(cfg.objective.scalar_weights.w_comm),
        "w_therm": float(cfg.objective.scalar_weights.w_therm),
        "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
    }
    evaluator = LayoutEvaluator(
        sigma_mm=float(cfg.objective.sigma_mm),
        baseline=baseline,
        scalar_w=scalar_w,
    )
    return evaluator, baseline


def _initial_assignments(layout_input: dict) -> Tuple[np.ndarray, np.ndarray]:
    assign_grid = np.array(layout_input["baseline"]["assign_grid"], dtype=int)
    assign_seed = np.array(layout_input["seed"]["assign_seed"], dtype=int)
    return assign_grid, assign_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    layout_input = load_layout_input(Path(args.layout_input))
    cfg = load_config(args.cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wafer_radius = float(layout_input["wafer"]["radius_mm"])
    sites_xy = np.array(layout_input["sites"]["sites_xy"], dtype=np.float32)
    chip_tdp = np.array(layout_input["slots"]["tdp"], dtype=float)
    traffic = np.array(layout_input["mapping"]["traffic_matrix"], dtype=float)
    traffic_sym = traffic + traffic.T
    S = int(layout_input["slots"]["S"])
    assign_grid, assign_seed_input = _initial_assignments(layout_input)

    evaluator, baseline = _build_evaluator(layout_input, cfg)
    pareto = ParetoSet(
        eps_comm=float(cfg.pareto.get("eps_comm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        eps_therm=float(cfg.pareto.get("eps_therm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        max_points=int(cfg.pareto.get("max_points", 2000)) if hasattr(cfg, "pareto") else 2000,
    )
    pareto_records: List[Dict] = []
    trace = TraceLogger(out_dir / "trace.csv")

    start_ts = time.time()
    seed_list = getattr(cfg.layout_agent, "seed_list", [0]) if hasattr(cfg, "layout_agent") else [0]
    best_assign = assign_seed_input

    for seed_id in seed_list:
        layout_state = LayoutState(
            S=S,
            Ns=sites_xy.shape[0],
            wafer_radius_mm=wafer_radius,
            sites_xy_mm=sites_xy,
            assign=assign_seed_input.copy(),
            chip_tdp_w=chip_tdp,
            traffic_bytes=traffic,
            meta={"seed": seed_id},
        )

        # Stage0 baseline and seed
        layout_state.assign = assign_grid.copy()
        evaluate_and_log(evaluator, layout_state, pareto, pareto_records, trace, stage="baseline", seed_id=seed_id, iter_idx=0)
        layout_state.assign = assign_seed_input.copy()
        evaluate_and_log(evaluator, layout_state, pareto, pareto_records, trace, stage="seed", seed_id=seed_id, iter_idx=0)

        # Stage1: coarsen
        clusters, W = coarsen_traffic(
            traffic_sym,
            slot_tdp=chip_tdp,
            target_num_clusters=int(cfg.coarsen.target_num_clusters),
            min_merge_traffic=float(cfg.coarsen.min_merge_traffic),
        )

        # Stage2: regions (optional)
        if getattr(cfg.regions, "enabled", True):
            regions, site_to_region = build_regions(
                sites_xy_mm=sites_xy,
                wafer_radius_mm=wafer_radius,
                ring_edges_ratio=cfg.regions.ring_edges_ratio,
                sectors_per_ring=cfg.regions.sectors_per_ring,
                ring_score=cfg.regions.ring_score,
                capacity_ratio=float(cfg.regions.capacity_ratio),
            )
            cluster_to_region, J = assign_clusters_to_regions(
                clusters,
                regions,
                W,
                lambda_graph=float(cfg.global_place_region.lambda_graph),
                lambda_ring=float(cfg.global_place_region.lambda_ring),
                lambda_cap=float(cfg.global_place_region.lambda_cap),
                refine_cfg=cfg.global_place_region.get("refine", {}),
            )
        else:
            regions = []
            site_to_region = np.zeros(sites_xy.shape[0], dtype=int)
            cluster_to_region = [0 for _ in clusters]
            J = 0.0

        # Stage3: expand
        assign_expand = expand_clusters(
            clusters=clusters,
            cluster_to_region=cluster_to_region,
            regions=regions,
            base_assign=np.full_like(assign_seed_input, -1),
            traffic_sym=traffic_sym,
            sites_xy=sites_xy,
            intra_refine_steps=int(cfg.expand.get("intra_refine_steps", 0)),
        )

        # Stage4: legalize
        assign_leg = legalize_assign(assign_expand, sites_xy, wafer_radius)
        layout_state.assign = assign_leg.copy()
        evaluate_and_log(evaluator, layout_state, pareto, pareto_records, trace, stage="legalize", seed_id=seed_id, iter_idx=0)

        # Stage5: detailed place (Pareto-aware SA + optional LLM)
        result = run_detailed_place(
            sites_xy=sites_xy,
            assign_seed=assign_leg,
            evaluator=evaluator,
            layout_state=layout_state,
            traffic_sym=traffic_sym,
            site_to_region=site_to_region,
            regions=regions,
            clusters=clusters,
            cluster_to_region=cluster_to_region,
            pareto=pareto,
            pareto_records=pareto_records,
            cfg=cfg.detailed_place if hasattr(cfg, "detailed_place") else {},
            trace_path=out_dir / f"trace_seed{seed_id}.csv",
            seed_id=seed_id,
            trace_logger=trace.log,
        )
        best_assign = result.assign

    # Stage6: alt-opt (optional)
    if hasattr(cfg, "alt_opt") and cfg.alt_opt.get("enabled", False):
        mapping_solver = MappingSolver(strategy="greedy_local", mem_limit_factor=1.0)
        best_assign, _ = run_alt_opt(
            rounds=int(cfg.alt_opt.get("rounds", 3)),
            mapping_solver=mapping_solver,
            segments=[],
            eff_specs={},
            sites_xy=sites_xy,
            assign_init=best_assign,
            evaluator=evaluator,
            layout_state=LayoutState(
                S=S,
                Ns=sites_xy.shape[0],
                wafer_radius_mm=wafer_radius,
                sites_xy_mm=sites_xy,
                assign=best_assign.copy(),
                chip_tdp_w=chip_tdp,
                traffic_bytes=traffic,
                meta={},
            ),
            pareto=pareto,
            cfg=cfg.alt_opt,
            trace_path=out_dir,
        )

    trace.close()
    _write_pareto_points(pareto_records, out_dir / "pareto_points.csv")
    best_comm, best_therm, best_payload = pareto.knee_point()
    runtime_sec = time.time() - start_ts
    layout_best = {
        "best": {
            "assign": best_payload.get("assign", best_assign).tolist(),
            "objectives": {"comm_norm": best_comm, "therm_norm": best_therm},
            "penalty": {},
            "meta": {"stage": best_payload.get("stage", "knee_point"), "seed_id": best_payload.get("seed_id", 0)},
        },
        "pareto_front": [
            {"comm_norm": p.comm_norm, "therm_norm": p.therm_norm, "payload": p.payload} for p in pareto.points
        ],
        "selection": {
            "method": cfg.pareto.get("selection", "knee_point_v1") if hasattr(cfg, "pareto") else "knee_point_v1",
            "pareto_size": len(pareto.points),
        },
        "region_plan": {"clusters": [c.slots for c in clusters], "cluster_to_region": cluster_to_region, "J": J},
        "artifacts": {
            "trace_csv": str((out_dir / "trace.csv").absolute()),
            "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
            "llm_usage_jsonl": str((out_dir / "llm_usage.jsonl").absolute()),
        },
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)

    report = {
        "baseline": {"comm_norm": 1.0, "therm_norm": 1.0},
        "knee": {"comm_norm": best_comm, "therm_norm": best_therm},
        "pareto_size": len(pareto.points),
        "runtime_sec": runtime_sec,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
