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
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

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
    "d_total",
    "d_comm",
    "d_therm",
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


def _signature_for_assign(assign: Iterable[int]) -> str:
    return "assign:" + ",".join(str(int(x)) for x in assign)


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


def _load_llm_config(llm_config_path: Path) -> None:
    if not llm_config_path.exists():
        return
    with llm_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    endpoint = str(cfg.get("endpoint", "")).strip()
    model = str(cfg.get("model", "")).strip()
    api_key = str(cfg.get("api_key", "")).strip()
    if endpoint:
        os.environ.setdefault("VOLC_ARK_ENDPOINT", endpoint)
    if model:
        os.environ.setdefault("VOLC_ARK_MODEL", model)
    if api_key and "REPLACE_WITH_API_KEY" not in api_key:
        os.environ.setdefault("VOLC_ARK_API_KEY", api_key)


def _build_evaluator(layout_input: dict, cfg: dict) -> LayoutEvaluator:
    return LayoutEvaluator(
        sigma_mm=float(cfg.objective.sigma_mm),
        baseline={
            "L_comm_baseline": float(layout_input.get("baseline", {}).get("L_comm", 1.0)),
            "L_therm_baseline": float(layout_input.get("baseline", {}).get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(cfg.objective.scalar_weights.w_comm),
            "w_therm": float(cfg.objective.scalar_weights.w_therm),
            "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
        },
    )


def _build_eval_assign(layout_input: dict, evaluator: LayoutEvaluator) -> Callable[[np.ndarray], Dict[str, Any]]:
    base_state = LayoutState(
        S=int(layout_input["slots"].get("S", len(layout_input["slots"]["tdp"]))),
        Ns=int(layout_input["sites"].get("Ns", len(layout_input["sites"]["sites_xy"]))),
        wafer_radius_mm=float(layout_input["wafer"]["radius_mm"]),
        sites_xy_mm=np.asarray(layout_input["sites"]["sites_xy"], dtype=np.float32),
        assign=np.zeros(int(layout_input["slots"].get("S", len(layout_input["slots"]["tdp"]))), dtype=int),
        chip_tdp_w=np.asarray(layout_input["slots"]["tdp"], dtype=float),
        traffic_bytes=np.asarray(layout_input["mapping"]["traffic_matrix"], dtype=float),
        meta={},
    )

    def _eval_assign(assign: np.ndarray) -> Dict[str, Any]:
        base_state.assign = np.asarray(assign, dtype=int)
        return evaluator.evaluate(base_state)

    return _eval_assign


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
) -> Dict[str, Any]:
    trace_path = out_dir / "trace.csv"
    best_eval: Dict[str, Any] | None = None
    best_assign: List[int] | None = None
    prev_eval: Dict[str, Any] | None = None
    accepts = 0

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

            operator = rec.get("operator")
            op_name = getattr(operator, "name", operator.__class__.__name__ if operator else "none")
            op_args = {}
            if operator is not None and hasattr(operator, "__dict__"):
                op_args = {k: v for k, v in operator.__dict__.items()}
            op_args_payload = {
                "op": op_name,
                **op_args,
                **(rec.get("meta") or {}),
            }
            signature = _signature_for_assign(assign)
            op_args_payload["signature"] = signature
            writer.writerow(
                [
                    idx,
                    rec.get("stage", "heuragenix"),
                    op_name,
                    json.dumps(op_args_payload),
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
                    d_total,
                    d_comm,
                    d_therm,
                ]
            )
        trace_fp.flush()

    return {
        "trace_path": trace_path,
        "best_eval": best_eval,
        "best_assign": best_assign,
        "accepts": accepts,
    }


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

    try:
        from pipeline.hyper_heuristics import LLMSelectionHyperHeuristic, RandomSelectionHyperHeuristic
        from problems.wafer_layout.env import Env
        from util.get_heuristic import get_heuristic
    except Exception as exc:  # noqa: BLE001
        _append_llm_usage(
            llm_usage_path,
            {
                "ok": False,
                "reason": "import_error",
                "error": str(exc),
            },
        )
        raise

    evaluator = _build_evaluator(layout_input, cfg)
    eval_assign = _build_eval_assign(layout_input, evaluator)

    env = Env(str(heuragenix_data_path), rng=rng, algorithm_data={"eval_assign": eval_assign})

    heuristic_dir_name = str(baseline_cfg.get("heuristic_dir", "basic_heuristics"))
    heuristic_dir = heuragenix_root / "src" / "problems" / "wafer_layout" / "heuristics" / heuristic_dir_name
    try:
        heuristics_map = get_heuristic(heuristic_dir_name, "wafer_layout")
        heuristics = list(heuristics_map.values())
    except Exception:  # noqa: BLE001
        heuristics = _import_heuristics(
            "problems.wafer_layout.heuristics." + heuristic_dir_name,
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
    fallback_method = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    llm_config_file = baseline_cfg.get("llm_config_file")
    if llm_config_file:
        llm_cfg_path = (heuragenix_root / str(llm_config_file)).resolve()
        _load_llm_config(llm_cfg_path)
    start_time = time.time()
    fallback_used = False

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
            stage_name="heuragenix_llm_hh",
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
            stage_name="heuragenix_random_hh",
        )
        remaining_steps = max_steps
        if env.recordings:
            remaining_steps = max(0, max_steps - len(env.recordings))
        random_hh.run(remaining_steps)
        for rec in _iter_selection_records(random_hh.usage_records, fallback_used):
            _append_llm_usage(llm_usage_path, rec)

    trace_info = _record_trace(out_dir, seed, env.recordings, eval_assign)

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

    report = {
        "best_total": float(best_eval.get("total_scalar", 0.0)),
        "best_comm": float(best_eval.get("comm_norm", 0.0)),
        "best_therm": float(best_eval.get("therm_norm", 0.0)),
        "steps_total": int(max_steps),
        "accepts_total": int(trace_info["accepts"]),
        "method": method,
        "fallback_used": fallback_used,
        "fallback_method": fallback_method,
        "runtime_s": float(time.time() - start_time),
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        "iterations_scale_factor": float(iterations_scale_factor),
        "selection_frequency": selection_frequency,
        "num_candidate_heuristics": num_candidate_heuristics,
        "rollout_budget": int(baseline_cfg.get("rollout_budget", 0)),
        **trace_metrics,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
