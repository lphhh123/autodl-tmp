"""Offline EDA-Agent driver (SPEC v5.4)."""
from __future__ import annotations

# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import csv
import hashlib
import json
import random
import time

import numpy as np

from layout.alt_opt import run_alt_opt
from layout.coarsen import coarsen_traffic
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.expand import expand_clusters
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.pareto_io import write_pareto_points_csv
from layout.regions import build_regions
from layout.sites import build_sites
from layout.trace_metrics import compute_trace_metrics_from_csv
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything


def load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_oscillation_metrics(trace_path: Path, window: int, eps_flat: float) -> dict:
    return compute_trace_metrics_from_csv(trace_path, window, eps_flat)


def _parse_segments(raw_segments) -> list[Segment]:
    segments: list[Segment] = []
    if not raw_segments:
        return segments
    for idx, seg in enumerate(raw_segments):
        segments.append(
            Segment(
                id=seg.get("id", idx),
                layer_ids=seg.get("layer_ids", []),
                flops=seg.get("flops", 0.0),
                bytes=seg.get("bytes", 0.0),
                seq_len=seg.get("seq_len", 0),
                embed_dim=seg.get("embed_dim", 0),
                num_heads=seg.get("num_heads", 0),
                mlp_ratio=seg.get("mlp_ratio", 0.0),
                precision=seg.get("precision", 1),
                traffic_in_bytes=seg.get("traffic_in_bytes", 0.0),
                traffic_out_bytes=seg.get("traffic_out_bytes", 0.0),
            )
        )
    return segments


def run_layout_agent(cfg, out_dir: Path, seed: int, layout_input_path: str | Path | None = None) -> None:
    if layout_input_path is None:
        layout_input_path = getattr(cfg, "layout_input", None)
    if not layout_input_path:
        raise RuntimeError("Missing layout_input: please set cfg.layout_input or pass layout_input_path")
    layout_input = load_layout_input(Path(layout_input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_usage_path = out_dir / "llm_usage.jsonl"
    llm_usage_path.parent.mkdir(parents=True, exist_ok=True)
    if not llm_usage_path.exists():
        llm_usage_path.write_text("", encoding="utf-8")

    start_time = time.time()
    seed_everything(int(seed))

    wafer_radius = float(layout_input["wafer"]["radius_mm"])
    sites_xy = np.array(layout_input["sites"]["sites_xy"], dtype=np.float32)
    assign_grid = np.array(layout_input["baseline"]["assign_grid"], dtype=int)
    assign_seed = np.array(layout_input["seed"]["assign_seed"], dtype=int)
    chip_tdp = np.array(layout_input["slots"]["tdp"], dtype=float)
    traffic = np.array(layout_input["mapping"]["traffic_matrix"], dtype=float)
    traffic_sym = traffic + traffic.T
    mapping_current = layout_input.get("mapping", {}).get("mapping", list(range(assign_seed.shape[0])))
    segments = _parse_segments(layout_input.get("mapping", {}).get("segments", []))
    S = assign_grid.shape[0]
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
        Ns=sites_xy.shape[0],
        wafer_radius_mm=wafer_radius,
        sites_xy_mm=sites_xy,
        assign=assign_seed.copy(),
        chip_tdp_w=chip_tdp,
        traffic_bytes=traffic,
        meta={"stage": "seed"},
    )
    mapping_solver = MappingSolver(
        strategy=str(getattr(getattr(cfg, "alt_opt", {}), "mapping_strategy", "greedy_local")),
        mem_limit_factor=float(getattr(getattr(cfg, "alt_opt", {}), "mem_limit_factor", 1.0)),
    )
    pareto = ParetoSet(
        eps_comm=float(cfg.pareto.get("eps_comm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        eps_therm=float(cfg.pareto.get("eps_therm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        max_points=int(cfg.pareto.get("max_points", 2000)) if hasattr(cfg, "pareto") else 2000,
    )

    # Stage0: baseline evaluations
    layout_state.assign = assign_grid
    base_eval = evaluator.evaluate(layout_state)
    base_added = pareto.add(
        base_eval["comm_norm"],
        base_eval["therm_norm"],
        {"assign": assign_grid.copy(), "total_scalar": base_eval["total_scalar"], "stage": "baseline", "iter": 0, "seed": -1},
    )
    layout_state.assign = assign_seed
    seed_eval = evaluator.evaluate(layout_state)
    seed_added = pareto.add(
        seed_eval["comm_norm"],
        seed_eval["therm_norm"],
        {"assign": assign_seed.copy(), "total_scalar": seed_eval["total_scalar"], "stage": "seed", "iter": 0, "seed": 0},
    )

    # Stage1: coarsen
    clusters, W = coarsen_traffic(
        traffic_sym,
        slot_tdp=chip_tdp,
        target_num_clusters=int(cfg.coarsen.target_num_clusters),
        min_merge_traffic=float(cfg.coarsen.min_merge_traffic),
    )

    # Stage2: regions
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
        py_rng=random.Random(int(seed)),
    )

    # Stage3: expand
    assign_expand = expand_clusters(
        clusters=clusters,
        cluster_to_region=cluster_to_region,
        regions=regions,
        base_assign=np.full_like(assign_seed, -1),
        traffic_sym=traffic_sym,
        sites_xy=sites_xy,
        intra_refine_steps=int(cfg.expand.get("intra_refine_steps", 0)),
    )

    # Stage4: legalize
    assign_leg = legalize_assign(assign_expand, sites_xy, wafer_radius)
    layout_state.assign = assign_leg

    # Stage5: detailed place
    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
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
        cfg=detailed_cfg,
        trace_path=out_dir / "trace.csv",
        seed_id=int(seed),
        chip_tdp=chip_tdp,
        llm_usage_path=out_dir / "llm_usage.jsonl",
    )
    assign_final = result.assign

    # Stage6: alt-opt (optional)
    mapping_final = layout_input["mapping"].get("mapping") if "mapping" in layout_input else None
    if hasattr(cfg, "alt_opt") and cfg.alt_opt.get("enabled", False):
        assign_final, mapping_final = run_alt_opt(
            rounds=int(cfg.alt_opt.get("rounds", 3)),
            mapping_solver=mapping_solver,
            segments=segments,
            mapping_init=mapping_current,
            traffic_sym=traffic_sym,
            sites_xy=sites_xy,
            assign_init=assign_final,
            evaluator=evaluator,
            layout_state=layout_state,
            pareto=pareto,
            cfg=cfg.alt_opt,
            trace_path=out_dir / "trace.csv",
            chip_tdp=chip_tdp,
        )

    # Outputs
    write_pareto_points_csv(pareto, out_dir / "pareto_points.csv")
    best_comm, best_therm, best_payload = pareto.knee_point()
    best_assign = best_payload.get("assign", assign_final)
    layout_state.assign = np.array(best_assign, dtype=int)
    best_eval = evaluator.evaluate(layout_state)
    runtime_s = time.time() - start_time
    layout_best = {
        "best": {
            "assign": best_assign.tolist(),
            "pos_xy_mm": sites_xy[best_assign].tolist(),
            "objectives": {"comm_norm": best_comm, "therm_norm": best_therm},
            "raw": {"L_comm": best_eval["L_comm"], "L_therm": best_eval["L_therm"]},
            "penalty": best_eval["penalty"],
            "meta": {"stage": "knee_point"},
        },
        "pareto_front": [
            {"comm_norm": p.comm_norm, "therm_norm": p.therm_norm} for p in pareto.points
        ],
        "selection": {
            "method": cfg.pareto.get("selection", "knee_point_v1") if hasattr(cfg, "pareto") else "knee_point_v1",
            "pareto_size": len(pareto.points),
        },
        "region_plan": {
            "clusters": [c.slots for c in clusters],
            "cluster_to_region": cluster_to_region,
            "J": J,
        },
        "artifacts": {
            "trace_csv": str((out_dir / "trace.csv").absolute()),
            "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
            "llm_usage_jsonl": str((out_dir / "llm_usage.jsonl").absolute()),
        },
        "alt_opt": {
            "enabled": bool(cfg.alt_opt.get("enabled", False)) if hasattr(cfg, "alt_opt") else False,
            "mapping_final": mapping_final,
        },
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)

    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = compute_trace_metrics_from_csv(out_dir / "trace.csv", metrics_window, eps_flat)

    report = {
        "baseline": {"comm_norm": base_eval["comm_norm"], "therm_norm": base_eval["therm_norm"]},
        "knee": {"comm_norm": best_comm, "therm_norm": best_therm},
        "best_total": float(best_eval.get("total_scalar", 0.0)),
        "best_comm": float(best_eval.get("comm_norm", best_comm)),
        "best_therm": float(best_eval.get("therm_norm", best_therm)),
        "pareto_size": len(pareto.points),
        "pareto_front_size": len(pareto.points),
        "alt_opt_rounds": int(cfg.alt_opt.get("rounds", 0)) if hasattr(cfg, "alt_opt") else 0,
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        "runtime_s": float(runtime_s),
        "evaluator_calls": int(getattr(evaluator, "evaluator_calls", getattr(evaluator, "evaluate_calls", 0))),
        "evaluate_calls": int(getattr(evaluator, "evaluate_calls", 0)),
        "policy_switch": result.policy_meta.get("policy_switch") if result.policy_meta else None,
        "cache": result.policy_meta.get("cache") if result.policy_meta else None,
        **trace_metrics,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/layout_agent/layout_L0_heuristic.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout_input", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="layout")
    seed_everything(int(args.seed))

    # auto out_dir if not provided
    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/layout_agent") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sync cfg.train.out_dir
    if hasattr(cfg, "train"):
        cfg.train.out_dir = str(out_dir)

    # dump config_used
    try:
        (out_dir / "config_used.yaml").write_text(Path(args.cfg).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # dump resolved config
    try:
        import omegaconf
        (out_dir / "config_resolved.yaml").write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(str(cfg), encoding="utf-8")

    layout_input_path = args.layout_input or getattr(cfg, "layout_input", None)
    layout_hash = None
    if layout_input_path:
        try:
            layout_hash = hashlib.sha256(Path(layout_input_path).read_bytes()).hexdigest()
        except Exception:
            layout_hash = None
    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    extra = {
        "repo_root": str(_PROJECT_ROOT),
        "problem_name": "wafer_layout",
        "layout_input_hash": layout_hash,
        "steps": detailed_cfg.get("max_steps") if isinstance(detailed_cfg, dict) else None,
        "budget": detailed_cfg.get("budget") if isinstance(detailed_cfg, dict) else None,
    }
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    from utils.stable_hash import stable_hash

    cfg_hash = stable_hash({"cfg": resolved_text})
    if hasattr(cfg, "train"):
        cfg.train.cfg_hash = cfg_hash
        cfg.train.cfg_path = str(args.cfg)
    try:
        from utils.run_manifest import write_run_manifest

        write_run_manifest(
            out_dir=str(out_dir),
            cfg_path=str(args.cfg),
            cfg_hash=str(cfg_hash),
            seed=int(args.seed),
            stable_hw_state={},
            extra=extra,
        )
    except Exception:
        pass

    # run meta
    meta = {"argv": sys.argv, "out_dir": str(out_dir), "cfg_path": str(args.cfg), "seed": int(args.seed)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    run_layout_agent(cfg, out_dir=out_dir, seed=int(args.seed), layout_input_path=args.layout_input)


if __name__ == "__main__":
    main()
