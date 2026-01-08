"""HeurAgenix baseline wrapper for wafer layout (SPEC v4.3.2/4)."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from layout.candidate_pool import _signature_for_action, inverse_signature
from layout.evaluator import LayoutEvaluator, LayoutState
from scripts.run_layout_agent import compute_oscillation_metrics
from utils.config import load_config


TRACE_FIELDS = [
    "iter",
    "stage",
    "op",
    "op_args_json",
    "accepted",
    "total_scalar",
    "comm_norm",
    "therm_norm",
    "pareto_added",
    "duplicate_penalty",
    "boundary_penalty",
    "seed_id",
    "time_ms",
    "signature",
    "inverse_signature",
    "d_total",
    "d_comm",
    "d_therm",
    "tabu_hit",
    "inverse_hit",
    "cooldown_hit",
]


def _load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_heuragenix_root(cfg_root: str | None, project_root: Path) -> Path:
    if cfg_root:
        candidate = Path(cfg_root).expanduser()
        if candidate.exists():
            return candidate
    fallback = project_root / "HeurAgenix"
    return fallback


def _copy_layout_input(layout_input_path: Path, heuragenix_root: Path) -> Path:
    case_name = layout_input_path.stem
    target_dir = heuragenix_root / "data" / "wafer_layout" / "data" / "test_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{case_name}.json"
    shutil.copyfile(layout_input_path, target_path)
    return target_path


def _import_heuristics(module_prefix: str, heuristic_dir: Path) -> List[Callable[..., Any]]:
    heuristics: List[Callable[..., Any]] = []
    for py_file in sorted(heuristic_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        name = py_file.stem
        module = importlib.import_module(f"{module_prefix}.{name}")
        if not hasattr(module, name):
            raise AttributeError(f"Heuristic {name} missing in {module_prefix}.{name}")
        heuristics.append(getattr(module, name))
    if not heuristics:
        raise RuntimeError(f"No heuristics found under {heuristic_dir}")
    return heuristics


def _parse_layout_input(layout_input: dict, cfg: dict) -> tuple[LayoutState, LayoutEvaluator, np.ndarray, Callable[[np.ndarray], Dict[str, Any]]]:
    wafer = layout_input.get("wafer", {})
    slots = layout_input.get("slots", {})
    sites = layout_input.get("sites", {})
    baseline = layout_input.get("baseline", {})
    seed = layout_input.get("seed", {})

    sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
    tdp = np.asarray(slots.get("tdp", []), dtype=float)
    S = int(slots.get("S", len(tdp)))
    Ns = int(sites.get("Ns", len(sites_xy)))
    traffic = np.asarray(layout_input.get("mapping", {}).get("traffic_matrix", []), dtype=float)
    if traffic.size == 0:
        traffic = np.zeros((S, S), dtype=float)

    init_assign = None
    if seed.get("assign_seed") is not None:
        init_assign = np.asarray(seed.get("assign_seed"), dtype=int)
    if init_assign is None and seed.get("seed_assign") is not None:
        init_assign = np.asarray(seed.get("seed_assign"), dtype=int)
    if init_assign is None and baseline.get("assign_grid") is not None:
        init_assign = np.asarray(baseline.get("assign_grid"), dtype=int)
    if init_assign is None and layout_input.get("baseline_assign_grid") is not None:
        init_assign = np.asarray(layout_input.get("baseline_assign_grid"), dtype=int)
    if init_assign is None:
        init_assign = np.arange(S, dtype=int) % max(1, Ns)

    evaluator = LayoutEvaluator(
        sigma_mm=float(cfg.objective.sigma_mm),
        baseline={
            "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
            "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(cfg.objective.scalar_weights.w_comm),
            "w_therm": float(cfg.objective.scalar_weights.w_therm),
            "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
        },
    )

    base_state = LayoutState(
        S=S,
        Ns=Ns,
        wafer_radius_mm=float(wafer.get("radius_mm", 0.0)),
        sites_xy_mm=sites_xy,
        assign=np.asarray(init_assign, dtype=int).copy(),
        chip_tdp_w=tdp,
        traffic_bytes=traffic,
        meta={"margin_mm": float(wafer.get("margin_mm", 0.0))},
    )

    def _eval_assign(assign: np.ndarray) -> Dict[str, Any]:
        base_state.assign = np.asarray(assign, dtype=int)
        out = evaluator.evaluate(base_state)
        penalty = out.get("penalty", {})
        out["penalty_norm"] = float(penalty.get("duplicate", 0.0) + penalty.get("boundary", 0.0))
        return out

    return base_state, evaluator, np.asarray(init_assign, dtype=int), _eval_assign


def _operator_to_action(operator: Any, meta: Dict[str, Any], pre_assign: np.ndarray) -> Dict[str, Any]:
    if operator is None:
        return {"op": "do_nothing", "type": "do_nothing"}
    op_name = getattr(operator, "name", operator.__class__.__name__)
    action: Dict[str, Any] = {"op": op_name, "type": op_name}
    if op_name == "swap" and hasattr(operator, "i") and hasattr(operator, "j"):
        action.update({"i": int(operator.i), "j": int(operator.j), "type": "swap"})
    elif op_name == "relocate" and hasattr(operator, "i") and hasattr(operator, "site_id"):
        slot = int(operator.i)
        from_site = int(pre_assign[slot]) if 0 <= slot < len(pre_assign) else None
        action.update(
            {
                "i": slot,
                "site_id": int(operator.site_id),
                "from_site": from_site,
                "type": "relocate",
            }
        )
    elif op_name == "cluster_move":
        action.update(
            {
                "cluster_id": int(meta.get("cluster_id", -1)),
                "region_id": int(meta.get("region_id", -1)),
                "target_sites": getattr(operator, "target_sites", None),
                "slots": getattr(operator, "slots", None),
                "type": "cluster_move",
            }
        )
    elif op_name == "random_kick":
        ops = getattr(operator, "ops", []) or []
        action.update({"k": len(ops), "type": "random_kick"})
    elif op_name in {"noop", "do_nothing"}:
        action.update({"op": "do_nothing", "type": "do_nothing"})
    action.update(meta or {})
    return action


def _append_llm_usage(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _iter_selection_records(records: List[Dict[str, Any]], fallback_used: bool) -> Iterable[Dict[str, Any]]:
    for rec in records:
        payload = dict(rec)
        payload.setdefault("ok", True)
        payload.setdefault("fallback_used", fallback_used)
        yield payload


def _record_trace(
    out_dir: Path,
    seed: int,
    recordings: List[Dict[str, Any]],
    eval_assign: Callable[[np.ndarray], Dict[str, Any]],
    init_assign: np.ndarray,
) -> Dict[str, Any]:
    trace_path = out_dir / "trace.csv"
    best_eval: Dict[str, Any] | None = None
    best_assign: List[int] | None = None
    prev_eval: Dict[str, Any] | None = None
    accepts = 0
    best_step = -1
    prev_assign = np.asarray(init_assign, dtype=int).copy()

    with trace_path.open("w", encoding="utf-8", newline="") as trace_fp:
        writer = csv.writer(trace_fp)
        writer.writerow(TRACE_FIELDS)
        for idx, rec in enumerate(recordings):
            assign = rec.get("assign")
            if assign is None:
                continue
            eval_out = eval_assign(np.asarray(assign, dtype=int))
            if prev_eval is None:
                prev_eval = eval_out
            d_total = float(eval_out["total_scalar"] - prev_eval["total_scalar"])
            d_comm = float(eval_out["comm_norm"] - prev_eval["comm_norm"])
            d_therm = float(eval_out["therm_norm"] - prev_eval["therm_norm"])
            prev_eval = eval_out

            accepted = int(bool(rec.get("accepted")))
            if accepted:
                accepts += 1
            if best_eval is None or eval_out["total_scalar"] < best_eval["total_scalar"]:
                best_eval = dict(eval_out)
                best_assign = list(assign)
                best_step = idx

            operator = rec.get("operator")
            meta = rec.get("meta") or {}
            action = _operator_to_action(operator, meta, prev_assign)
            op_name = action.get("op", "none")
            signature = _signature_for_action(action, prev_assign)
            inv_signature = inverse_signature(action, prev_assign)
            writer.writerow(
                [
                    idx,
                    rec.get("stage", "heuragenix"),
                    op_name,
                    json.dumps(action, ensure_ascii=False, sort_keys=True),
                    accepted,
                    float(eval_out["total_scalar"]),
                    float(eval_out["comm_norm"]),
                    float(eval_out["therm_norm"]),
                    0,
                    float(eval_out["penalty"]["duplicate"]),
                    float(eval_out["penalty"]["boundary"]),
                    seed,
                    int(rec.get("time_ms", 0)),
                    signature,
                    inv_signature,
                    d_total,
                    d_comm,
                    d_therm,
                    0,
                    0,
                    0,
                ]
            )
            prev_assign = np.asarray(assign, dtype=int).copy()
        trace_fp.flush()

    return {
        "trace_path": trace_path,
        "best_eval": best_eval,
        "best_assign": best_assign,
        "accepts": accepts,
        "best_step": best_step,
    }


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return {k: _cfg_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_cfg_to_dict(v) for v in cfg]
    if hasattr(cfg, "__dict__") and not isinstance(cfg, (str, int, float, bool)):
        return {k: _cfg_to_dict(v) for k, v in cfg.__dict__.items()}
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    layout_input_path = Path(args.layout_input)
    layout_input = _load_layout_input(layout_input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_usage_path = out_dir / "llm_usage.jsonl"
    llm_usage_path.touch(exist_ok=True)

    seed = int(args.seed)
    _seed_everything(seed)
    rng = random.Random(seed)

    project_root = Path(__file__).resolve().parents[1]
    baseline_cfg = cfg.baseline if hasattr(cfg, "baseline") else {}
    heuragenix_root = _resolve_heuragenix_root(baseline_cfg.get("heuragenix_root"), project_root)
    heuragenix_data_path = _copy_layout_input(layout_input_path, heuragenix_root)

    sys.path.insert(0, str(heuragenix_root))
    sys.path.insert(0, str(heuragenix_root / "src"))
    sys.path.insert(0, str(project_root))

    from pipeline.hyper_heuristics import LLMSelectionHyperHeuristic, RandomSelectionHyperHeuristic
    from problems.wafer_layout.env import Env
    from util.llm_client.get_llm_client import get_llm_client

    base_state, evaluator, init_assign, eval_assign = _parse_layout_input(layout_input, cfg)

    env = Env(str(heuragenix_data_path), rng=rng, algorithm_data={"eval_assign": eval_assign})

    heuristic_dir = heuragenix_root / "src" / "problems" / "wafer_layout" / "heuristics" / str(
        baseline_cfg.get("heuristic_dir", "basic_heuristics")
    )
    heuristics = _import_heuristics(
        "problems.wafer_layout.heuristics." + str(baseline_cfg.get("heuristic_dir", "basic_heuristics")),
        heuristic_dir,
    )

    S = int(layout_input["slots"].get("S", len(layout_input["slots"]["tdp"])))
    max_steps = int(baseline_cfg.get("max_steps", 0))
    iterations_scale_factor = float(baseline_cfg.get("iterations_scale_factor", 1.0))
    if max_steps > 0:
        iterations_scale_factor = max_steps / max(1, S)
    else:
        max_steps = max(1, int(iterations_scale_factor * max(1, S)))

    selection_frequency = int(baseline_cfg.get("selection_frequency", 5))
    num_candidate_heuristics = int(baseline_cfg.get("num_candidate_heuristics", 4))
    sa_T0 = float(baseline_cfg.get("sa_T0", 1.0))
    sa_alpha = float(baseline_cfg.get("sa_alpha", 0.995))

    method = str(baseline_cfg.get("method", "random_hh"))
    start_time = time.time()
    fallback_used = False
    fallback_method = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    llm_client = None
    llm_config_file = baseline_cfg.get("llm_config_file")
    if llm_config_file:
        llm_client = get_llm_client(
            str(Path(llm_config_file).expanduser()),
            timeout_sec=int(baseline_cfg.get("llm_timeout_s", 30)),
            max_retry=int(baseline_cfg.get("max_llm_failures", 1)),
        )

    if method == "llm_hh":
        llm_timeout = int(baseline_cfg.get("llm_timeout_s", 30))
        llm_max_retry = int(baseline_cfg.get("max_llm_failures", 1))
        llm_hh = LLMSelectionHyperHeuristic(
            env,
            heuristics,
            rng,
            selection_frequency=selection_frequency,
            num_candidate_heuristics=num_candidate_heuristics,
            sa_T0=sa_T0,
            sa_alpha=sa_alpha,
            timeout_sec=llm_timeout,
            max_retry=llm_max_retry,
            llm_client=llm_client,
        )
        try:
            llm_hh.run(max_steps)
            for rec in _iter_selection_records(llm_hh.selection_records, False):
                _append_llm_usage(llm_usage_path, rec)
        except Exception as exc:  # noqa: BLE001
            _append_llm_usage(
                llm_usage_path,
                {
                    "ok": False,
                    "reason": "llm_failure",
                    "error": str(exc),
                    "fallback_used": True,
                    "method": method,
                },
            )
            fallback_used = True

    if method != "llm_hh" or fallback_used:
        random_hh = RandomSelectionHyperHeuristic(
            env,
            heuristics,
            rng,
            selection_frequency=selection_frequency,
            sa_T0=sa_T0,
            sa_alpha=sa_alpha,
        )
        remaining_steps = max_steps
        if env.recordings:
            remaining_steps = max(0, max_steps - len(env.recordings))
        random_hh.run(remaining_steps)
        for rec in _iter_selection_records(random_hh.usage_records, fallback_used):
            _append_llm_usage(llm_usage_path, rec)
        if method != "llm_hh":
            _append_llm_usage(llm_usage_path, {"ok": False, "mode": "random_hh"})

    trace_info = _record_trace(out_dir, seed, env.recordings, eval_assign, init_assign)

    detailed_cfg = cfg.get("detailed_place", {})
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = compute_oscillation_metrics(trace_info["trace_path"], metrics_window, eps_flat)

    best_eval = trace_info["best_eval"] or {}
    best_assign = trace_info["best_assign"] or env.solution.assign

    layout_best = {
        "assign": list(best_assign),
        "best_total": float(best_eval.get("total_scalar", 0.0)),
        "best_comm": float(best_eval.get("comm_norm", 0.0)),
        "best_therm": float(best_eval.get("therm_norm", 0.0)),
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)

    llm_report = {
        "ok": bool(method == "llm_hh" and not fallback_used),
        "provider": getattr(llm_client, "provider", None) if llm_client else None,
        "model": getattr(llm_client, "model", None) if llm_client else None,
        "llm_config_file": str(llm_config_file) if llm_config_file else None,
        "total_calls": len(getattr(llm_client, "calls", [])) if llm_client else None,
        "fallback": fallback_method if fallback_used else None,
    }
    report = {
        "cfg_path": str(args.cfg),
        "cfg_dump": _cfg_to_dict(cfg),
        "best_total": float(best_eval.get("total_scalar", 0.0)),
        "best_comm": float(best_eval.get("comm_norm", 0.0)),
        "best_therm": float(best_eval.get("therm_norm", 0.0)),
        "steps_total": int(max_steps),
        "best_step": int(trace_info.get("best_step", -1)),
        "accepts_total": int(trace_info["accepts"]),
        "method": method,
        "fallback_used": fallback_used,
        "fallback_method": fallback_method,
        "runtime_s": float(time.time() - start_time),
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        "iterations_scale_factor": float(iterations_scale_factor),
        "llm": llm_report,
        "oscillation_metrics": trace_metrics,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
