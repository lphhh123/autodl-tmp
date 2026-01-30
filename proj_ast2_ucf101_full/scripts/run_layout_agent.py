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
import shutil
import time
import uuid
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from layout.alt_opt import run_alt_opt
from layout.candidate_pool import signature_from_assign
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
from utils.config_utils import get_nested
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.seed import seed_everything
from utils.stable_hash import stable_hash
from utils.trace_guard import (
    init_trace_dir_v54,
    append_trace_event_v54,
    finalize_trace_dir,
    update_trace_summary,
)
from utils.trace_contract_v54 import TraceContractV54
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS


def load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_oscillation_metrics(trace_path: Path, window: int, eps_flat: float) -> dict:
    return compute_trace_metrics_from_csv(trace_path, window, eps_flat)


def _is_valid_assign(assign: list[int] | np.ndarray | None, S: int, Ns: int) -> bool:
    if assign is None:
        return False
    if len(assign) != int(S):
        return False
    return all(0 <= int(x) < int(Ns) for x in assign)


def _append_trace_events_from_csv(trace_events_path: Path, trace_csv: Path, run_id: str) -> int:
    if not trace_csv.exists():
        return 0
    added = 0
    with trace_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = str(row.get("stage", "") or "")
            if stage in {"init", "finalize"}:
                continue

            def _num(val, default=0.0):
                try:
                    if val is None or val == "":
                        return default
                    return float(val)
                except Exception:
                    return default

            def _int(val, default=0):
                try:
                    if val is None or val == "":
                        return default
                    return int(float(val))
                except Exception:
                    return default

            iter_id = _int(row.get("iter", 0), 0)
            comm_norm = _num(row.get("comm_norm", 0.0), 0.0)
            therm_norm = _num(row.get("therm_norm", 0.0), 0.0)

            append_trace_event_v54(
                trace_events_path,
                "layout_step",
                payload={
                    "iter": int(iter_id),
                    "stage": stage,
                    "op": str(row.get("op", "")),
                    "op_args_json": str(row.get("op_args_json", "")),
                    "accepted": _int(row.get("accepted", 0), 0),
                    "total_scalar": _num(row.get("total_scalar", 0.0), 0.0),
                    "comm_norm": comm_norm,
                    "therm_norm": therm_norm,
                    "pareto_added": _int(row.get("pareto_added", 0), 0),
                    "duplicate_penalty": _num(row.get("duplicate_penalty", 0.0), 0.0),
                    "boundary_penalty": _num(row.get("boundary_penalty", 0.0), 0.0),
                    "seed_id": _int(row.get("seed_id", 0), 0),
                    "time_ms": _int(row.get("time_ms", 0.0), 0),
                    "signature": str(row.get("signature", "")),
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            append_trace_event_v54(
                trace_events_path,
                "proxy_sanitize",
                payload={
                    "candidate_id": int(iter_id),
                    "outer_iter": int(iter_id),
                    "inner_step": int(iter_id),
                    "metric": "comm",
                    "raw_value": float(comm_norm),
                    "used_value": float(comm_norm),
                    "penalty_added": 0.0,
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            append_trace_event_v54(
                trace_events_path,
                "proxy_sanitize",
                payload={
                    "candidate_id": int(iter_id),
                    "outer_iter": int(iter_id),
                    "inner_step": int(iter_id),
                    "metric": "therm",
                    "raw_value": float(therm_norm),
                    "used_value": float(therm_norm),
                    "penalty_added": 0.0,
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            added += 1
    return added

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


def run_layout_agent(
    cfg,
    out_dir: Path,
    seed: int,
    layout_input_path: str | Path | None = None,
    run_id: str | None = None,
) -> None:
    if layout_input_path is None:
        layout_input_path = getattr(cfg, "layout_input", None)
    if not layout_input_path:
        raise RuntimeError("Missing layout_input: please set cfg.layout_input or pass layout_input_path")
    layout_input = load_layout_input(Path(layout_input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out = Path(getattr(cfg, "output_dir", out_dir))
    base_out.mkdir(parents=True, exist_ok=True)
    # ---- v5.4: canonical trace events (JSONL) ----
    if run_id is None:
        run_id = stable_hash(
            {
                "mode": "layout_agent",
                "cfg_path": str(getattr(cfg, "cfg_path", "")),
                "seed_id": int(seed),
                "layout_input": str(layout_input_path),
            }
        )
    sig = build_signature_v54(
        cfg,
        method_name="ours_layout_agent",
        overrides={"seed_global": int(seed), "seed_problem": int(seed)},
    )
    meta = init_trace_dir_v54(
        base_dir=base_out,
        run_id=run_id,
        cfg=cfg,
        signature=sig,
        signature_v54=sig,
        run_meta={
            "heuristic": "ours_layout_agent",
            "layout_input": str(layout_input_path),
            "mode": "layout",
            "run_id": run_id,
            "seed_id": int(seed),
            "requested": {"method": "layout_agent"},
            "effective": {"method": "layout_agent"},
            "seal_check_policy": {"kind": "per_step", "interval": 1},
        },
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
    )
    trace_dir = meta["trace_dir"]
    out_dir = trace_dir
    llm_usage_path = trace_dir / "llm_usage.jsonl"
    llm_usage_path_compat = base_out / "llm_usage.jsonl"
    if not llm_usage_path.exists():
        llm_usage_path.write_text("", encoding="utf-8")
    trace_path = trace_dir / "trace.csv"
    trace_path_compat = base_out / "trace.csv"
    trace_events_path = trace_dir / "trace_events.jsonl"
    with (trace_dir / "trace_header.json").open("r", encoding="utf-8") as f:
        header_payload = json.load(f)
    append_trace_event_v54(
        trace_events_path=trace_events_path,
        run_id=run_id,
        step=0,
        event_type="trace_header",
        payload=header_payload,
        strict=True,
    )
    trace_contract = TraceContractV54
    seal_digest = str(get_nested(cfg, "contract.seal_digest", "") or "")
    assert_cfg_sealed_or_violate(
        cfg=cfg,
        seal_digest=seal_digest,
        trace_events_path=trace_events_path,
        phase="layout_agent",
        step=0,
        fatal=True,
        trace_contract=trace_contract,
    )

    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    planned_steps = int(detailed_cfg.get("max_steps", detailed_cfg.get("steps", 0)) or 0)
    # ---- budget (SPEC_D/E): evaluator-calls hard budget ----
    budget_cfg = getattr(cfg, "budget", None) or {}
    total_eval_budget = int(getattr(budget_cfg, "total_eval_budget", 0) or 0)
    max_wallclock_sec = float(getattr(budget_cfg, "max_wallclock_sec", 0.0) or 0.0)
    search_eval_budget = total_eval_budget - 1 if total_eval_budget and total_eval_budget > 1 else total_eval_budget

    start_time = time.time()
    seed_everything(int(seed))
    best_assign = None
    best_total = None
    evaluator = None
    S = 0
    Ns = 0
    ok = False
    try:
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
        Ns = sites_xy.shape[0]
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
        class _BudgetExceeded(RuntimeError):
            pass

        _orig_eval = evaluator.evaluate

        def _budgeted_evaluate(state):
            if search_eval_budget and evaluator.evaluator_calls >= search_eval_budget:
                raise _BudgetExceeded("eval budget exhausted")
            return _orig_eval(state)

        evaluator.evaluate = _budgeted_evaluate
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
        pareto.add(
            base_eval["comm_norm"],
            base_eval["therm_norm"],
            {"assign": assign_grid.copy(), "total_scalar": base_eval["total_scalar"], "stage": "baseline", "iter": 0, "seed": -1},
        )
        layout_state.assign = assign_seed
        seed_eval = evaluator.evaluate(layout_state)
        pareto.add(
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
        budget_exhausted = False
        try:
            assert_cfg_sealed_or_violate(
                cfg=cfg,
                seal_digest=seal_digest,
                trace_events_path=trace_events_path,
                phase="layout_agent.before_detailed_place",
                step=1,
                fatal=True,
                trace_contract=trace_contract,
            )
            # IMPORTANT(v5.4): pass the FULL sealed cfg into detailed_place, not cfg.detailed_place.
            # detailed_place will read its parameters from cfg.detailed_place internally.
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
                cfg=cfg,
                trace_path=trace_path,
                seed_id=int(seed),
                chip_tdp=chip_tdp,
                llm_usage_path=llm_usage_path,
                recordings_path=out_dir / "recordings.jsonl",
                trace_events_path=trace_events_path,
            )
            assert_cfg_sealed_or_violate(
                cfg=cfg,
                seal_digest=seal_digest,
                trace_events_path=trace_events_path,
                phase="layout_agent.after_detailed_place",
                step=2,
                fatal=True,
                trace_contract=trace_contract,
            )
        except _BudgetExceeded:
            budget_exhausted = True
            _, _, payload = pareto.knee_point()
            best_assign = payload.get("assign", None)
            best_total = payload.get("total_scalar", None)
            result = SimpleNamespace(assign=assign_leg, policy_meta={})
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
                trace_path=trace_path,
                chip_tdp=chip_tdp,
            )

        # Outputs
        write_pareto_points_csv(pareto, out_dir / "pareto_points.csv")
        best_comm, best_therm, best_payload = pareto.knee_point()
        best_assign = best_payload.get("assign", assign_final)
        layout_state.assign = np.array(best_assign, dtype=int)
        best_eval = None
        if best_assign is not None and (not total_eval_budget or evaluator.evaluator_calls < total_eval_budget):
            evaluator.evaluate = _orig_eval
            best_eval = evaluator.evaluate(layout_state)
        runtime_s = time.time() - start_time
        best_total = float(best_eval.get("total_scalar", 0.0)) if best_eval else float(best_total or 0.0)
        layout_best = {
            "run_id": run_id,
            "best": {
                "assign": best_assign.tolist(),
                "pos_xy_mm": sites_xy[best_assign].tolist(),
                "objectives": {"comm_norm": best_comm, "therm_norm": best_therm},
                "raw": {
                    "L_comm": best_eval["L_comm"] if best_eval else None,
                    "L_therm": best_eval["L_therm"] if best_eval else None,
                },
                "penalty": best_eval["penalty"] if best_eval else {},
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
                "trace_csv": str(trace_path.absolute()),
                "trace_csv_compat": str(trace_path_compat.absolute()),
                "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
                "llm_usage_jsonl": str(llm_usage_path.absolute()),
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
        trace_metrics = compute_trace_metrics_from_csv(trace_path, metrics_window, eps_flat)

        report = {
            "run_id": run_id,
            "baseline": {"comm_norm": base_eval["comm_norm"], "therm_norm": base_eval["therm_norm"]},
            "knee": {"comm_norm": best_comm, "therm_norm": best_therm},
            "best_total": float(best_eval.get("total_scalar", 0.0)) if best_eval else float(best_total or 0.0),
            "best_comm": float(best_eval.get("comm_norm", best_comm)) if best_eval else float(best_comm),
            "best_therm": float(best_eval.get("therm_norm", best_therm)) if best_eval else float(best_therm),
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

        # ---- v5.4: enforce budget fairness artifact (budget.json) ----
        try:
            actual_calls = int(getattr(evaluator, "evaluator_calls", 0) if evaluator else 0)
            budget = {
                "run_id": run_id,
                "primary_limit": {"type": "eval_calls", "limit": int(total_eval_budget)},
                "secondary_limit": {"type": "wall_time_s", "limit": float(max_wallclock_sec)},
                "actual_eval_calls": actual_calls,
                "effective_eval_calls": actual_calls,
                "cache_hits_total": 0,
                "wall_time_s": float(time.time() - start_time),
                "planned_total_eval_budget": int(total_eval_budget),
                "planned_search_eval_budget": int(search_eval_budget),
                "seed_id": int(seed) if "seed" in locals() else None,
                "method": "ours",
                "budget_exhausted": bool(budget_exhausted),
                "actual_evaluator_calls": actual_calls,
            }
            (out_dir / "budget.json").write_text(
                json.dumps(budget, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            raise RuntimeError(f"failed to write budget.json: {exc}") from exc
        ok = True
    except Exception as exc:
        steps_done = int(getattr(evaluator, "evaluate_calls", 0)) if evaluator is not None else 0
        raise
    finally:
        steps_done = int(getattr(evaluator, "evaluate_calls", 0)) if evaluator is not None else 0
        best_solution_valid = _is_valid_assign(best_assign, S, Ns)
        trace_append_err = None
        try:
            _append_trace_events_from_csv(
                trace_events_path=trace_events_path,
                trace_csv=trace_path,
                run_id=str(run_id),
            )
        except Exception as e:
            trace_append_err = repr(e)
            ok = False
            raise
        finally:
            reason = "ok" if (ok and trace_append_err is None) else f"trace_append_error:{trace_append_err}"
            if trace_dir is not None:
                update_trace_summary(
                    trace_dir,
                    ok=bool(ok and trace_append_err is None),
                    reason=reason,
                    steps_done=int(planned_steps or steps_done),
                    best_solution_valid=bool(ok and best_assign is not None),
                )
                finalize_trace_dir(
                    trace_events_path,
                    reason=reason,
                    steps_done=int(planned_steps or steps_done),
                    best_solution_valid=bool(ok and best_assign is not None),
                )
            if trace_path.exists():
                shutil.copy2(trace_path, trace_path_compat)
            try:
                if llm_usage_path.exists():
                    llm_usage_path_compat.write_text(
                        llm_usage_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/layout_agent/layout_L0_heuristic.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout_input", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, False)
    # auto out_dir if not provided
    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/layout_agent") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex

    # sync cfg.train.out_dir
    if not hasattr(cfg, "train") or cfg.train is None:
        cfg.train = OmegaConf.create({})
    elif not OmegaConf.is_config(cfg.train):
        if isinstance(cfg.train, dict):
            cfg.train = OmegaConf.create(cfg.train)
        else:
            cfg.train = OmegaConf.create(dict(cfg.train))
    cfg.train.out_dir = str(out_dir)
    cfg.out_dir = str(out_dir)

    cfg = validate_and_fill_defaults(cfg, mode="layout")
    seed_everything(int(args.seed))

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
    layout_meta = {}
    if layout_input_path:
        try:
            layout_hash = hashlib.sha256(Path(layout_input_path).read_bytes()).hexdigest()
        except Exception:
            layout_hash = None
        try:
            layout_meta = load_layout_input(Path(layout_input_path)).get("meta", {}) or {}
        except Exception:
            layout_meta = {}
    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    extra_meta = {
        "budget_main_axis": "eval_calls",
        "dataset_id": f"wafer_layout:{layout_meta.get('layout_id', 'unknown')}",
    }
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")

    cfg_hash = stable_hash({"cfg": resolved_text})
    from utils.run_manifest import write_run_manifest

    write_run_manifest(
        out_dir=str(out_dir),
        cfg_path=str(args.cfg),
        cfg_hash=str(cfg_hash),
        seed=int(args.seed),
        stable_hw_state={
            "guard_mode": "acc_first_hard_gating",
            "lambda_hw_base": None,
            "lambda_hw_effective": None,
            "discrete_cache": {
                "mapping_signature": str(meta.get("mapping_signature", "")) if "meta" in locals() else "",
                "layout_signature": str(meta.get("layout_signature", "")) if "meta" in locals() else "",
            },
        },
        cfg=cfg,
        extra=extra_meta,
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )

    # run meta
    meta = {
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "cfg_path": str(args.cfg),
        "cfg_hash": str(cfg_hash),
        "seed": int(args.seed),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    run_layout_agent(
        cfg,
        out_dir=out_dir,
        seed=int(args.seed),
        layout_input_path=args.layout_input,
        run_id=run_id,
    )
    layout_best_path = out_dir / "layout_best.json"
    layout_best = json.loads(layout_best_path.read_text(encoding="utf-8")) if layout_best_path.exists() else {}
    best_assign = layout_best.get("best", {}).get("assign", []) if isinstance(layout_best, dict) else []
    meta = {
        "mapping_signature": "",
        "layout_signature": signature_from_assign(best_assign) if best_assign else "",
    }

    write_run_manifest(
        out_dir=str(out_dir),
        cfg_path=str(args.cfg),
        cfg_hash=str(cfg_hash),
        seed=int(args.seed),
        stable_hw_state={
            "guard_mode": "acc_first_hard_gating",
            "lambda_hw_base": None,
            "lambda_hw_effective": None,
            "discrete_cache": {
                "mapping_signature": str(meta.get("mapping_signature", "")) if "meta" in locals() else "",
                "layout_signature": str(meta.get("layout_signature", "")) if "meta" in locals() else "",
            },
        },
        cfg=cfg,
        extra=extra_meta,
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
