"""HeurAgenix baseline wrapper for wafer layout (SPEC v4.3.2/4)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from layout.candidate_pool import _signature_for_action
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from scripts.run_layout_agent import _write_pareto_points, compute_oscillation_metrics
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
        if not candidate.is_absolute():
            for base in (project_root, project_root.parent):
                adjusted = base / candidate
                if adjusted.exists():
                    return adjusted
        if candidate.exists():
            return candidate
    fallback = project_root / "HeurAgenix"
    if fallback.exists():
        return fallback
    sibling = project_root.parent / "HeurAgenix"
    return sibling


def _resolve_llm_config_path(llm_config_file: str | None, heuragenix_root: Path, project_root: Path) -> Path | None:
    if not llm_config_file:
        return None
    candidate = Path(llm_config_file).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (heuragenix_root, project_root):
        path = base / candidate
        if path.exists():
            return path
    return None


def _ensure_prompt_files(target_root: Path, heuragenix_root: Path) -> None:
    prompt_dir = target_root / "data" / "wafer_layout" / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        heuragenix_root / "data" / "wafer_layout" / "prompt",
        heuragenix_root / "src" / "problems" / "wafer_layout" / "prompt",
    ]
    source = next((c for c in candidates if c.exists()), None)
    if source is None:
        return
    for file in source.glob("*.txt"):
        target = prompt_dir / file.name
        if not target.exists():
            shutil.copyfile(file, target)


def _build_objective_cfg(cfg: Any) -> Dict[str, Any]:
    objective = cfg.objective if hasattr(cfg, "objective") else {}
    scalar = objective.get("scalar_weights", {}) if isinstance(objective, dict) else objective.scalar_weights
    return {
        "sigma_mm": float(objective.get("sigma_mm", 20.0)) if isinstance(objective, dict) else float(objective.sigma_mm),
        "scalar_weights": {
            "w_comm": float(scalar.get("w_comm", 0.7)) if isinstance(scalar, dict) else float(scalar.w_comm),
            "w_therm": float(scalar.get("w_therm", 0.3)) if isinstance(scalar, dict) else float(scalar.w_therm),
            "w_penalty": float(scalar.get("w_penalty", 1000.0)) if isinstance(scalar, dict) else float(scalar.w_penalty),
        },
    }


def _write_test_data(layout_input_path: Path, heuragenix_root: Path, problem: str, seed: int) -> Tuple[str, Path]:
    # MUST match HeurAgenix launch_hyper_heuristic base_dir/data layout
    data_dir = heuragenix_root / "data" / problem / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"case_seed{seed}"
    target = data_dir / f"{case_name}.json"
    layout_input = json.loads(layout_input_path.read_text(encoding="utf-8"))
    target.write_text(json.dumps(layout_input, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_name, target


def _prepare_work_dir(
    out_dir: Path,
    heuragenix_root: Path,
    layout_input_path: Path,
    problem: str,
    seed: int,
) -> Tuple[Path, str, Path]:
    work_dir = out_dir / "__heuragenix_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_prompt_files(heuragenix_root, heuragenix_root)
    case_name, case_path = _write_test_data(layout_input_path, heuragenix_root, problem, seed)
    return work_dir, case_name, case_path


def infer_problem_size(layout_input: dict) -> int:
    chiplets = layout_input.get("chiplets", None)
    if isinstance(chiplets, list) and len(chiplets) > 0:
        return len(chiplets)
    if "S" in layout_input:
        return int(layout_input["S"])
    slots = layout_input.get("slots", None)
    if isinstance(slots, dict) and "S" in slots:
        return int(slots["S"])
    return 1


def _resolve_heuristic_files(heuragenix_root: Path, problem: str, heuristic_dir: str) -> List[str]:
    candidate = Path(heuristic_dir)
    if not candidate.is_absolute():
        candidate = heuragenix_root / "src" / "problems" / problem / "heuristics" / heuristic_dir
    if not candidate.exists():
        raise FileNotFoundError(f"Heuristic directory not found: {candidate}")
    files = [str(path) for path in sorted(candidate.glob("*.py")) if not path.name.startswith("__")]
    if not files:
        raise RuntimeError(f"No heuristics found in {candidate}")
    return files


def _derive_initial_assign(layout_input: dict) -> np.ndarray:
    baseline = layout_input.get("baseline", {})
    seed = layout_input.get("seed", {})
    slots = layout_input.get("slots", {})
    tdp = np.asarray(slots.get("tdp", []), dtype=int)
    S = int(slots.get("S", len(tdp)))
    sites_xy = np.asarray(layout_input.get("sites", {}).get("sites_xy", []), dtype=float)
    Ns = int(layout_input.get("sites", {}).get("Ns", len(sites_xy)))

    init_assign = None
    if seed.get("assign_seed") is not None:
        init_assign = np.asarray(seed.get("assign_seed"), dtype=int)
    elif seed.get("seed_assign") is not None:
        init_assign = np.asarray(seed.get("seed_assign"), dtype=int)
    elif baseline.get("assign_grid") is not None:
        init_assign = np.asarray(baseline.get("assign_grid"), dtype=int)
    elif layout_input.get("baseline_assign_grid") is not None:
        init_assign = np.asarray(layout_input.get("baseline_assign_grid"), dtype=int)
    if init_assign is None:
        if Ns <= 0:
            init_assign = np.zeros(S, dtype=int)
        else:
            init_assign = np.arange(S, dtype=int) % max(1, Ns)
    return init_assign


def _build_evaluator(layout_input: dict, cfg: Any) -> Tuple[LayoutState, LayoutEvaluator]:
    wafer = layout_input.get("wafer", {})
    slots = layout_input.get("slots", {})
    sites = layout_input.get("sites", {})
    baseline = layout_input.get("baseline", {})

    sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
    tdp = np.asarray(slots.get("tdp", []), dtype=float)
    S = int(slots.get("S", len(tdp)))
    Ns = int(len(sites_xy))
    traffic = np.asarray(layout_input.get("mapping", {}).get("traffic_matrix", []), dtype=float)
    if traffic.size == 0:
        traffic = np.zeros((S, S), dtype=float)

    objective_cfg = _build_objective_cfg(cfg)
    evaluator = LayoutEvaluator(
        sigma_mm=float(objective_cfg["sigma_mm"]),
        baseline={
            "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
            "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(objective_cfg["scalar_weights"]["w_comm"]),
            "w_therm": float(objective_cfg["scalar_weights"]["w_therm"]),
            "w_penalty": float(objective_cfg["scalar_weights"]["w_penalty"]),
        },
    )

    init_assign = _derive_initial_assign(layout_input)
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
    return base_state, evaluator


def _iter_recordings(recordings_path: Path) -> Iterable[Dict[str, Any]]:
    with recordings_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _action_from_record(record: Dict[str, Any], prev_assign: np.ndarray) -> Dict[str, Any]:
    op = record.get("op", "noop")
    op_args = record.get("op_args", {}) or {}
    action = {"op": op, "type": op}
    if op == "swap":
        action.update({"i": int(op_args.get("i", -1)), "j": int(op_args.get("j", -1))})
    elif op == "relocate":
        slot = int(op_args.get("i", -1))
        from_site = op_args.get("from_site")
        if from_site is None and 0 <= slot < len(prev_assign):
            from_site = int(prev_assign[slot])
        action.update({"i": slot, "site_id": int(op_args.get("site_id", -1)), "from_site": from_site})
    elif op == "cluster_move":
        action.update(
            {
                "slots": op_args.get("slots"),
                "target_sites": op_args.get("target_sites"),
                "cluster_id": op_args.get("cluster_id", -1),
                "region_id": op_args.get("region_id", -1),
                "from_region": op_args.get("from_region"),
            }
        )
    elif op == "random_kick":
        action.update({"k": op_args.get("k")})
    action.setdefault("from_region", op_args.get("from_region"))
    return action


def _write_trace_and_pareto(
    out_dir: Path,
    seed: int,
    recordings_path: Path,
    layout_input: dict,
    cfg: Any,
    max_steps: int | None,
) -> Tuple[Dict[str, Any], ParetoSet]:
    base_state, evaluator = _build_evaluator(layout_input, cfg)
    pareto_cfg = cfg.get("pareto", {}) if isinstance(cfg, dict) else cfg.pareto
    pareto = ParetoSet(
        eps_comm=float(pareto_cfg.get("eps_comm", 0.0)),
        eps_therm=float(pareto_cfg.get("eps_therm", 0.0)),
        max_points=int(pareto_cfg.get("max_points", 2000)),
    )
    trace_path = out_dir / "trace.csv"
    prev_metrics = evaluator.evaluate(base_state)
    prev_total = float(prev_metrics["total_scalar"])
    prev_comm = float(prev_metrics["comm_norm"])
    prev_therm = float(prev_metrics["therm_norm"])
    prev_assign = _derive_initial_assign(layout_input)
    best_eval: Dict[str, Any] | None = None
    best_assign: List[int] | None = None
    accepts = 0
    best_step = -1

    with trace_path.open("w", encoding="utf-8", newline="") as f_trace:
        writer = csv.writer(f_trace)
        writer.writerow(TRACE_FIELDS)
        for idx, rec in enumerate(_iter_recordings(recordings_path)):
            if max_steps is not None and idx >= max_steps:
                break
            if "assign" in rec and rec["assign"] is not None:
                base_state.assign = np.asarray(rec["assign"], dtype=int)
            assign = base_state.assign
            eval_out = evaluator.evaluate(base_state)
            d_total = float(eval_out["total_scalar"] - prev_total)
            d_comm = float(eval_out["comm_norm"] - prev_comm)
            d_therm = float(eval_out["therm_norm"] - prev_therm)

            accepted = int(bool(rec.get("accepted", True)))
            if accepted:
                accepts += 1
            if best_eval is None or eval_out["total_scalar"] < best_eval.get("total_scalar", float("inf")):
                best_eval = dict(eval_out)
                best_assign = list(assign)
                best_step = idx

            action = _action_from_record(rec, prev_assign)
            signature = rec.get("signature") or _signature_for_action(
                {"op": rec.get("op"), **(rec.get("op_args") or {})},
                prev_assign,
            )
            pareto_added = pareto.add(
                eval_out["comm_norm"],
                eval_out["therm_norm"],
                {
                    "assign": assign.copy(),
                    "total_scalar": eval_out["total_scalar"],
                    "stage": "heuragenix",
                    "iter": idx,
                    "seed": seed,
                },
            )
            writer.writerow(
                [
                    idx,
                    "heuragenix",
                    action.get("op", "noop"),
                    json.dumps(action, ensure_ascii=False, sort_keys=True),
                    accepted,
                    float(eval_out["total_scalar"]),
                    float(eval_out["comm_norm"]),
                    float(eval_out["therm_norm"]),
                    int(pareto_added),
                    float(eval_out["penalty"]["duplicate"]),
                    float(eval_out["penalty"]["boundary"]),
                    seed,
                    int(rec.get("time_ms", 0)),
                    signature,
                    d_total,
                    d_comm,
                    d_therm,
                    0,
                    0,
                    0,
                ]
            )
            prev_assign = assign.copy()
            prev_total = float(eval_out["total_scalar"])
            prev_comm = float(eval_out["comm_norm"])
            prev_therm = float(eval_out["therm_norm"])
    return {
        "trace_path": trace_path,
        "best_eval": best_eval,
        "best_assign": best_assign,
        "accepts": accepts,
        "best_step": best_step,
        "steps_total": idx + 1 if "idx" in locals() else 0,
    }, pareto


def _write_layout_best(
    out_dir: Path,
    layout_input: dict,
    cfg: Any,
    pareto: ParetoSet,
    best_assign: List[int],
) -> Dict[str, Any]:
    base_state, evaluator = _build_evaluator(layout_input, cfg)
    base_state.assign = np.asarray(best_assign, dtype=int)
    eval_out = evaluator.evaluate(base_state)
    sites_xy = np.asarray(layout_input.get("sites", {}).get("sites_xy", []), dtype=float)
    pos_xy = sites_xy[best_assign].tolist() if len(best_assign) and len(sites_xy) else []
    best_comm, best_therm, _ = pareto.knee_point()
    layout_best = {
        "best": {
            "assign": list(best_assign),
            "pos_xy_mm": pos_xy,
            "objectives": {"comm_norm": float(eval_out.get("comm_norm", 0.0)), "therm_norm": float(eval_out.get("therm_norm", 0.0))},
            "raw": {"L_comm": eval_out.get("L_comm", 0.0), "L_therm": eval_out.get("L_therm", 0.0)},
            "penalty": eval_out.get("penalty", {}),
            "meta": {"stage": "heuragenix_best"},
        },
        "pareto_front": [{"comm_norm": p.comm_norm, "therm_norm": p.therm_norm} for p in pareto.points],
        "selection": {"method": "knee_point_v1", "pareto_size": len(pareto.points)},
        "knee_point": {"comm_norm": best_comm, "therm_norm": best_therm},
        "artifacts": {
            "trace_csv": str((out_dir / "trace.csv").absolute()),
            "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
            "llm_usage_jsonl": str((out_dir / "llm_usage.jsonl").absolute()),
        },
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)
    return layout_best


def _read_best_assign(best_solution_path: Path, fallback_assign: List[int]) -> List[int]:
    if not best_solution_path.exists():
        return fallback_assign
    payload = json.loads(best_solution_path.read_text(encoding="utf-8"))
    return payload.get("best_assign") or payload.get("best", {}).get("assign") or fallback_assign


def _append_llm_usage(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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

    seed = int(args.seed)
    _seed_everything(seed)

    project_root = Path(__file__).resolve().parents[1]
    baseline_cfg = cfg.baseline if hasattr(cfg, "baseline") else cfg.get("baseline", {})
    heuragenix_root = _resolve_heuragenix_root(baseline_cfg.get("heuragenix_root"), project_root)
    if not heuragenix_root.exists():
        raise FileNotFoundError(f"HeurAgenix root not found: {heuragenix_root}")

    method = str(baseline_cfg.get("method", "llm_hh"))
    fallback_method = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    start_time = time.time()
    llm_usage_path = out_dir / "llm_usage.jsonl"

    problem = str(baseline_cfg.get("problem", "wafer_layout"))
    work_dir, case_name, case_path = _prepare_work_dir(
        out_dir,
        heuragenix_root,
        layout_input_path,
        problem,
        seed,
    )
    heuristic_dir = str(baseline_cfg.get("heuristic_dir", "basic_heuristics"))
    selection_frequency = int(baseline_cfg.get("selection_frequency", 5))
    num_candidate_heuristics = int(baseline_cfg.get("num_candidate_heuristics", 4))
    rollout_budget = int(baseline_cfg.get("rollout_budget", 0))
    iters_sf = float(baseline_cfg.get("iterations_scale_factor", 2.0))
    max_steps = baseline_cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)
        S = infer_problem_size(layout_input)
        iters_sf = max(1.0, math.ceil(float(max_steps) / max(1, S)))

    output_root = heuragenix_root / "output" / problem

    fallback_used = False
    log_text = ""
    llm_config = None
    if method == "llm_hh":
        llm_config = _resolve_llm_config_path(baseline_cfg.get("llm_config_file"), heuragenix_root, project_root)
    output_dir = output_root / f"seed{seed}_{method}"

    launch_cmd = [
        sys.executable,
        str(heuragenix_root / "launch_hyper_heuristic.py"),
        "-p",
        problem,
        "-e",
        method,
        "-d",
        heuristic_dir,
        "-t",
        case_name,
        "-n",
        f"{iters_sf}",
        "-m",
        str(selection_frequency),
        "-c",
        str(num_candidate_heuristics),
        "-b",
        str(rollout_budget),
        "-r",
        "result",
        "--seed",
        str(seed),
    ]
    if method == "llm_hh" and llm_config is not None:
        launch_cmd.extend(["-l", str(llm_config)])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), str(heuragenix_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["AMLT_OUTPUT_DIR"] = str(heuragenix_root / "output")
    result = subprocess.run(
        launch_cmd,
        cwd=str(heuragenix_root),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        log_text = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
        _append_llm_usage(
            llm_usage_path,
            {"ok": False, "reason": "launch_failed", "engine": method, "error": log_text},
        )
        if method != fallback_method:
            fallback_used = True
            method = fallback_method
            output_dir = output_root / f"seed{seed}_{method}"
            launch_cmd = [
                sys.executable,
                str(heuragenix_root / "launch_hyper_heuristic.py"),
                "-p",
                problem,
                "-e",
                method,
                "-d",
                heuristic_dir,
                "-t",
                case_name,
                "-n",
                f"{iters_sf}",
                "-m",
                str(selection_frequency),
                "-c",
                str(num_candidate_heuristics),
                "-b",
                str(rollout_budget),
                "-r",
                "result",
                "--seed",
                str(seed),
            ]
            result = subprocess.run(
                launch_cmd,
                cwd=str(heuragenix_root),
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                log_text = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
                _append_llm_usage(
                    llm_usage_path,
                    {"ok": False, "reason": "fallback_launch_failed", "engine": method, "error": log_text},
                )

    cands = list(output_root.glob(f"**/seed{seed}_{method}/recordings.jsonl"))
    if not cands:
        cands = list(output_root.glob(f"**/seed{seed}_*/recordings.jsonl"))
    if cands:
        recordings_path = cands[0]
        output_dir = recordings_path.parent
        method = output_dir.name.split(f"seed{seed}_", 1)[-1]
        fallback_used = method != baseline_cfg.get("method", "llm_hh")
    else:
        output_dir = out_dir
        recordings_path = out_dir / "recordings.jsonl"
        _append_llm_usage(
            llm_usage_path,
            {"ok": False, "reason": "missing_recordings", "engine": method},
        )
        recordings_path.parent.mkdir(parents=True, exist_ok=True)
        recordings_path.write_text("", encoding="utf-8")
    heuragenix_usage_path = output_dir / "llm_usage.jsonl"
    if heuragenix_usage_path.exists() and heuragenix_usage_path.stat().st_size > 0:
        llm_usage_path.parent.mkdir(parents=True, exist_ok=True)
        with llm_usage_path.open("a", encoding="utf-8") as f:
            for line in heuragenix_usage_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    f.write(line + "\n")
    else:
        _append_llm_usage(
            llm_usage_path,
            {"ok": False, "reason": "missing_llm_usage", "engine": method},
        )
    trace_info, pareto = _write_trace_and_pareto(out_dir, seed, recordings_path, layout_input, cfg, max_steps)
    _write_pareto_points(pareto, out_dir / "pareto_points.csv")

    best_solution_path = output_dir / "best_solution.json"
    best_assign = _read_best_assign(best_solution_path, trace_info.get("best_assign") or _derive_initial_assign(layout_input).tolist())
    _write_layout_best(out_dir, layout_input, cfg, pareto, best_assign)

    detailed_cfg = cfg.get("detailed_place", {}) if isinstance(cfg, dict) else cfg.detailed_place
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = compute_oscillation_metrics(out_dir / "trace.csv", metrics_window, eps_flat)

    best_eval = trace_info.get("best_eval") or {}
    report = {
        "cfg_path": str(args.cfg),
        "best_total_scalar": float(best_eval.get("total_scalar", 0.0)),
        "best_comm_norm": float(best_eval.get("comm_norm", 0.0)),
        "best_therm_norm": float(best_eval.get("therm_norm", 0.0)),
        "best_total": float(best_eval.get("total_scalar", 0.0)),
        "steps_total": int(trace_info.get("steps_total", 0)),
        "accepts_total": int(trace_info.get("accepts", 0)),
        "best_step": int(trace_info.get("best_step", -1)),
        "method": method,
        "fallback_used": fallback_used,
        "fallback_method": fallback_method if fallback_used else None,
        "runtime_s": float(time.time() - start_time),
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        "iterations_scale_factor": float(iters_sf),
        "oscillation_metrics": trace_metrics,
        **trace_metrics,
        "raw_dir": str(output_dir),
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
