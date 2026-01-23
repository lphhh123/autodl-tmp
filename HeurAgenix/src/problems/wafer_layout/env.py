from __future__ import annotations
import json
import math
import os
import time
import random as _random
from typing import Any, Dict

import numpy as np

from src.problems.base.env import BaseEnv
from src.problems.wafer_layout.components import WaferLayoutSolution
from src.problems.wafer_layout.problem_state import (
    get_instance_problem_state,
    get_solution_problem_state,
    get_observation_problem_state,
)


def _load_layout_evaluator(self):
    import sys
    from pathlib import Path

    # 1) explicit override
    override = os.environ.get("PROJECT_EVAL_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        pkg_root = candidate / "proj_ast2_ucf101_full"
        if (pkg_root / "hw_proxy").exists():
            sys.path.insert(0, str(candidate))
            from layout.evaluator import LayoutEvaluator
            return LayoutEvaluator

    # 2) auto-discover from HeurAgenix repo root upward
    here = Path(__file__).resolve()
    heur_repo = here
    while heur_repo.name != "HeurAgenix" and heur_repo.parent != heur_repo:
        heur_repo = heur_repo.parent
    candidates = [
        heur_repo.parent,                   # sibling layout
        heur_repo.parent.parent,             # one more up
        Path.cwd(),
        Path.cwd().parent,
    ]
    for base in candidates:
        pkg = (base / "proj_ast2_ucf101_full")
        if (pkg / "hw_proxy").exists():
            sys.path.insert(0, str(base.resolve()))
            from layout.evaluator import LayoutEvaluator
            return LayoutEvaluator

    raise RuntimeError(
        "Cannot locate 'proj_ast2_ucf101_full' evaluator package. "
        "Set env PROJECT_EVAL_ROOT=<path containing proj_ast2_ucf101_full/> "
        "or run from a workspace where it is a sibling of HeurAgenix."
    )


def _assign_signature(assign_list: list[int]) -> str:
    from src.problems.wafer_layout.candidate_pool import signature_from_assign

    return signature_from_assign(assign_list)


def _op_signature_from_action(op: str, op_args: dict | None) -> str:
    from src.problems.wafer_layout.candidate_pool import op_signature

    args = op_args or {}
    i = int(args.get("i", -1)) if isinstance(args, dict) else -1
    site_id = int(args.get("site_id", args.get("j", -1))) if isinstance(args, dict) else -1
    candidate_id = int(args.get("candidate_id", -1)) if isinstance(args, dict) else -1
    return op_signature(op=op, i=i, site_id=site_id, candidate_id=candidate_id)


def _finite(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        raise ValueError(f"non-finite objective: {x}")
    return float(x)


class Env(BaseEnv):
    def load_data(self, data_path: str) -> dict:
        with open(data_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert "slots" in d and "sites" in d and "mapping" in d, f"bad instance: {data_path}"
        return d

    def __init__(self, data_name: str):
        super().__init__(data_name)

        self.problem = "wafer_layout"
        self._step_id = 0
        self._stage = "llm_hh"
        seed_info = self.instance_data.get("seed", {}) or {}
        assign_seed = seed_info.get("assign_seed", 0)
        if isinstance(assign_seed, int):
            self._seed_id = int(assign_seed)
        else:
            self._seed_id = int(seed_info.get("seed_id", 0))
        self.seed = int(self._seed_id)
        _random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = _random.Random(self._seed_id)

        slots = self.instance_data.get("slots", {})
        sites = self.instance_data.get("sites", {})
        self.S = int(slots.get("S", len(slots.get("tdp", slots.get("slot_tdp_w", [])))))
        self.Ns = int(sites.get("Ns", len(sites.get("sites_xy", []))))
        self.problem_size = self.S
        self.construction_steps = self.S

        obj = self.instance_data.get("objective_cfg", {}) or self.instance_data.get("objective", {})
        self.sigma_mm = float(obj.get("sigma_mm", 20.0))
        w = obj.get("scalar_weights", {}) or {}
        self.w_comm = float(w.get("w_comm", 0.7))
        self.w_therm = float(w.get("w_therm", 0.3))
        self.w_penalty = float(w.get("w_penalty", w.get("w_dup", 0.1) + w.get("w_boundary", 0.2)))

        base = self.instance_data.get("baseline", {}) or {}
        self.base_comm = float(base.get("L_comm", 1.0)) if base.get("L_comm", 0.0) else 1.0
        self.base_therm = float(base.get("L_therm", 1.0)) if base.get("L_therm", 0.0) else 1.0

        self.temp_init = float(obj.get("temp_init", 0.05))
        self.temp_min = float(obj.get("temp_min", 0.005))
        self.temp_decay = float(obj.get("temp_decay", 0.999))
        self._sa_T = float(self.temp_init)

        self._rec_fp = None
        self._rec_path = None
        self._best_path = None

        self._require_main_evaluator = bool(self.instance_data.get("require_main_evaluator", True))
        self._allow_fallback_evaluator = bool(self.instance_data.get("allow_fallback_evaluator", False))
        if bool(self.instance_data.get("force_main_evaluator", False)):
            self._require_main_evaluator = True

        self._evaluator_source = "main_project"
        self._evaluator_import_error = ""
        self._evaluator = None
        self._layout_state = None
        self._init_evaluator()
        self._max_eval_calls = int(self.instance_data.get("max_eval_calls", 0) or 0)
        self._meta_base = {
            "evaluator_source": self._evaluator_source,
            "evaluator_import_error": self._evaluator_import_error,
        }

        self.problem_state = self.get_problem_state()

    def _eval_assign(self, assign_list: list[int]) -> dict:
        if self._evaluator is None or self._layout_state is None:
            dup_count = len(assign_list) - len(set(assign_list))
            penalty_duplicate = float(dup_count) ** 2
            return {
                "total_scalar": float(penalty_duplicate),
                "comm_norm": 0.0,
                "therm_norm": 0.0,
                "penalty": {"duplicate": penalty_duplicate, "boundary": 0.0},
            }
        self._layout_state.assign = np.asarray(assign_list, dtype=np.int64)
        return self._evaluator.evaluate(self._layout_state)

    def _evaluate_assign(self, assign: list[int]) -> tuple[float, float, float]:
        metrics = self._eval_assign(assign)
        total_scalar = float(metrics.get("total_scalar", metrics.get("total", 0.0)))
        comm_norm = float(metrics.get("comm_norm", metrics.get("L_comm_norm", metrics.get("L_comm", 0.0))))
        therm_norm = float(metrics.get("therm_norm", metrics.get("L_therm_norm", metrics.get("L_therm", 0.0))))
        total_scalar = _finite(total_scalar)
        comm_norm = _finite(comm_norm)
        therm_norm = _finite(therm_norm)
        return total_scalar, comm_norm, therm_norm

    def _init_evaluator(self):
        try:
            LayoutEvaluator = _load_layout_evaluator(self)
            from layout.evaluator import LayoutState
            import_error = ""
        except Exception as exc:  # noqa: BLE001
            LayoutEvaluator = None
            LayoutState = None
            import_error = repr(exc)

        self._evaluator_import_error = import_error

        if LayoutEvaluator is None or LayoutState is None:
            if self._require_main_evaluator or (not self._allow_fallback_evaluator):
                raise RuntimeError(
                    "Failed to import main LayoutEvaluator (layout.evaluator). "
                    "Baseline requires main evaluator for comparability. "
                    "Hint: wafer_layout requires proj_ast2_ucf101_full on PYTHONPATH (layout.evaluator). "
                    f"import_error={import_error}"
                )
            self._evaluator = None
            self._layout_state = None
            self._evaluator_source = "fallback_penalty"
            return

        slots = self.instance_data.get("slots", {})
        sites = self.instance_data.get("sites", {})
        mapping = self.instance_data.get("mapping", {})
        wafer = self.instance_data.get("wafer", {})

        sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
        site_tdp_w = np.asarray(sites.get("site_tdp_w", np.zeros(len(sites_xy))), dtype=np.float32)
        traffic = np.asarray(mapping.get("traffic_matrix", mapping.get("traffic_bytes", [])), dtype=np.float32)
        slot_tdp = np.asarray(slots.get("tdp", slots.get("slot_tdp_w", [])), dtype=np.float32)
        if traffic.size == 0:
            traffic = np.zeros((self.S, self.S), dtype=np.float32)

        try:
            self._layout_state = LayoutState(
                S=int(self.S),
                Ns=int(self.Ns),
                wafer_radius_mm=float(wafer.get("radius_mm", 0.0)),
                sites_xy_mm=sites_xy,
                assign=np.zeros(self.S, dtype=np.int64),
                chip_tdp_w=slot_tdp,
                traffic_bytes=traffic,
                meta={"margin_mm": float(wafer.get("margin_mm", 0.0))},
            )
            self._evaluator = LayoutEvaluator(
                sigma_mm=self.sigma_mm,
                baseline={"L_comm_baseline": self.base_comm, "L_therm_baseline": self.base_therm},
                scalar_w={"w_comm": self.w_comm, "w_therm": self.w_therm, "w_penalty": self.w_penalty},
            )
            self._evaluator_source = "main_project"
        except Exception as exc:  # noqa: BLE001
            self._evaluator_import_error = f"{import_error}; init_error={repr(exc)}"
            if self._require_main_evaluator or (not self._allow_fallback_evaluator):
                raise RuntimeError(
                    "LayoutEvaluator import succeeded but initialization failed. "
                    "Baseline requires main evaluator for comparability. "
                    f"error={self._evaluator_import_error}"
                ) from exc
            self._evaluator = None
            self._layout_state = None
            self._evaluator_source = "fallback_penalty"

    def _sanitize_assign(self, assign: list[int]) -> list[int]:
        assign = list(assign or [])
        if len(assign) < self.S:
            assign.extend([0] * (self.S - len(assign)))
        assign = assign[: self.S]
        if self.Ns <= 0:
            return [0 for _ in range(self.S)]
        assign = [int(x) % self.Ns for x in assign]
        used = set()
        duplicate_idx = []
        for idx, value in enumerate(assign):
            if value in used:
                duplicate_idx.append(idx)
            else:
                used.add(value)
        free_sites = [i for i in range(self.Ns) if i not in used]
        for idx in duplicate_idx:
            if not free_sites:
                break
            assign[idx] = free_sites.pop()
        return assign

    def init_solution(self) -> WaferLayoutSolution:
        seed_obj = self.instance_data.get("seed", {}) or {}
        baseline = self.instance_data.get("baseline", {}) or {}

        if isinstance(seed_obj, dict) and seed_obj.get("assign_seed") is not None:
            assign = list(map(int, seed_obj["assign_seed"]))
        elif baseline.get("assign_grid") is not None:
            assign = list(map(int, baseline["assign_grid"]))
        else:
            if self.Ns >= self.S:
                assign = self.rng.sample(range(self.Ns), self.S)
            else:
                assign = [i % max(1, self.Ns) for i in range(self.S)]

        assign = self._sanitize_assign(assign)
        return WaferLayoutSolution(assign=assign)

    def is_complete_solution(self) -> bool:
        return True

    def validate_solution(self, solution=None) -> bool:
        sol = self.current_solution if solution is None else solution
        try:
            return (sol is not None) and hasattr(sol, "assign") and (len(sol.assign) == self.S)
        except Exception:
            return False

    @property
    def is_valid_solution(self) -> bool:
        return bool(self.validate_solution(self.current_solution))

    def compare(self, x, y):
        return y - x

    def get_key_value(self, solution: WaferLayoutSolution | None = None) -> float:
        sol = solution if solution is not None else self.current_solution
        total_scalar, _, _ = self._evaluate_assign(list(sol.assign))
        return float(total_scalar)

    def summarize_env(self) -> str:
        curr = float(self.key_value)
        best = float(getattr(self, "best_key_value", curr))
        return f"wafer_layout: S={self.S}, Ns={self.Ns}, step={self.current_steps}/{self.max_steps}, curr={curr:.6f}, best={best:.6f}"

    def get_problem_state(self) -> Dict[str, Any]:
        instance_state = get_instance_problem_state(self.instance_data)
        solution_state = get_solution_problem_state(self.instance_data, {"assign": list(self.current_solution.assign)})
        current_eval = dict(getattr(self, "current_eval", None) or self._eval_assign(list(self.current_solution.assign)))
        state = {
            "instance_data": self.instance_data,
            "current_solution": self.current_solution,
            "key_item": "total_scalar",
            "key_value": float(current_eval.get("total_scalar", 0.0)),
            "helper_function": {"eval_assign": self._eval_assign, "rng": self.rng},
            "instance_problem_state": instance_state,
            "solution_problem_state": solution_state,
            "instance": instance_state,
            "solution": solution_state,
        }
        state["observation_problem_state"] = get_observation_problem_state(state)
        return state

    def update_problem_state(self) -> None:
        self.problem_state = self.get_problem_state()

    def reset(self, output_dir: str = None):
        super().reset(output_dir=output_dir)
        self._step_id = 0
        self._sa_T = float(self.temp_init)
        ms = int(self.instance_data.get("max_steps", 0) or 0)
        self.max_steps = ms if ms > 0 else None
        seed_obj = self.instance_data.get("seed", {}) or {}
        seed = int(seed_obj.get("seed_id", getattr(self, "_seed_id", 0)) or 0)
        self.seed = seed
        self._seed_id = seed

        _random.seed(seed)
        np.random.seed(seed)
        self.rng = _random.Random(seed)

        self._rec_path = os.path.join(self.output_dir, "recordings.jsonl")
        self._best_path = os.path.join(self.output_dir, "best_solution.json")
        os.makedirs(self.output_dir, exist_ok=True)
        self._rec_fp = open(self._rec_path, "w", encoding="utf-8")

        self.current_eval = self._eval_assign(list(self.current_solution.assign))
        self.key_value = float(self.current_eval.get("total_scalar", 0.0))
        self.current_comm = float(self.current_eval.get("comm_norm", 0.0))
        self.current_therm = float(self.current_eval.get("therm_norm", 0.0))
        if self._evaluator is not None and hasattr(self._evaluator, "objective_hash"):
            self._objective_hash = self._evaluator.objective_hash()
        self.best_key_value = float(self.key_value)
        self.best_comm = float(self.current_comm)
        self.best_therm = float(self.current_therm)
        self.best_solution = WaferLayoutSolution(assign=list(self.current_solution.assign))

        self._dump_best()
        self.update_problem_state()
        try:
            penalty = self.current_eval.get("penalty", {}) if isinstance(self.current_eval.get("penalty", {}), dict) else {}
            duplicate_penalty = float(penalty.get("duplicate", 0.0))
            boundary_penalty = float(penalty.get("boundary", 0.0))
            assign_list = list(self.current_solution.assign) if hasattr(self.current_solution, "assign") else []

            init_record = {
                "iter": 0,
                "stage": "init",
                "op": "init",
                "op_args": {},
                "op_args_json": json.dumps({"op": "init"}, ensure_ascii=False),
                "assign": assign_list,
                "seed_id": int(self._seed_id),
                "accepted": 1,
                "total_scalar": float(self.key_value),
                "comm_norm": float(self.current_comm),
                "therm_norm": float(self.current_therm),
                "pareto_added": 0,
                "duplicate_penalty": duplicate_penalty,
                "boundary_penalty": boundary_penalty,
                "time_ms": 0,
                "eval_calls_cum": int(getattr(self._evaluator, "evaluate_calls", 0)),
                "cache_hit_cum": int(getattr(self._evaluator, "cache_hits", 0)) if hasattr(self._evaluator, "cache_hits") else 0,
                # v5.4 对齐：recordings.signature 统一为 assign signature（用于 repeat/oscillation 等指标）
                "signature": _assign_signature(assign_list),
                # 保留动作签名，便于动作级调试/回放
                "op_signature": _op_signature_from_action("init", {}),
                "meta": dict(self._meta_base),
            }
            self._write_record(init_record)
        except Exception:
            pass
        finally:
            # 关键：避免下一次 run_operator / run_heuristic 也写 iter=0
            # 即使 init_record 写失败，也必须推进 step_id
            self._step_id = 1

    def _dump_best(self):
        best = {
            "problem": "wafer_layout",
            "seed_id": int(self._seed_id),
            "best_assign": list(self.best_solution.assign),
            "best_key_value": float(self.best_key_value),
            "best_comm_norm": float(self.best_comm),
            "best_therm_norm": float(self.best_therm),
            "meta": {
                "S": self.S,
                "Ns": self.Ns,
                "radius_mm": self.instance_data.get("wafer", {}).get("radius_mm", 0.0),
                "weights": self.instance_data.get("objective_cfg", {}).get("scalar_weights", {}),
                "evaluator_source": self._evaluator_source,
                "evaluator_import_error": self._evaluator_import_error,
            },
        }
        with open(self._best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)

    def run_operator(
        self,
        operator,
        inplace: bool = True,
        meta: dict | None = None,
        stage: str = "heuragenix",
        time_ms: float | None = None,
        new_solution=None,
    ):
        t0 = time.time()
        old_assign = list(self.current_solution.assign)
        old_eval = dict(self.current_eval or {})
        old_key = float(old_eval.get("total_scalar", 0.0))
        old_comm = float(old_eval.get("comm_norm", 0.0))
        old_therm = float(old_eval.get("therm_norm", 0.0))

        if new_solution is None:
            operator.run(self.current_solution)
        else:
            self.current_solution = new_solution

        new_eval = self._eval_assign(list(self.current_solution.assign))
        new_key = float(new_eval.get("total_scalar", 0.0))
        new_comm = float(new_eval.get("comm_norm", 0.0))
        new_therm = float(new_eval.get("therm_norm", 0.0))

        delta = float(new_key - old_key)
        accept = bool(inplace) and (delta <= 0.0)
        if (not accept) and bool(inplace):
            T = float(getattr(self, "_sa_T", self.temp_init))
            if T > 1e-12:
                p = float(np.exp(-delta / max(T, 1e-12)))
                if self.rng.random() < p:
                    accept = True
            T = max(float(self.temp_min), float(T) * float(self.temp_decay))
            self._sa_T = float(T)
        if not accept:
            self.current_solution.assign = old_assign
            self.current_eval = old_eval
            new_key = old_key
            new_comm = old_comm
            new_therm = old_therm
        else:
            self.current_eval = new_eval

        self.key_value = float(new_key)
        self.current_comm = float(new_comm)
        self.current_therm = float(new_therm)
        if float(new_key) < float(self.best_key_value):
            self.best_key_value = float(new_key)
            self.best_comm = float(new_comm)
            self.best_therm = float(new_therm)
            self.best_solution = WaferLayoutSolution(assign=list(self.current_solution.assign))
            self._dump_best()

        self.update_problem_state()

        dt = (time.time() - t0) * 1000.0 if time_ms is None else float(time_ms)
        op_args_full = operator.to_action() if hasattr(operator, "to_action") else {"op": type(operator).__name__}
        op_name = op_args_full.get("op", type(operator).__name__)
        op_args = dict(op_args_full)
        op_args.pop("op", None)
        op_signature = _op_signature_from_action(op_name, op_args)
        assign_sig = _assign_signature(list(self.current_solution.assign))
        eval_for_record = self.current_eval if accept else old_eval
        penalty = eval_for_record.get("penalty", {}) if isinstance(eval_for_record.get("penalty", {}), dict) else {}
        duplicate_penalty = float(penalty.get("duplicate", 0.0))
        boundary_penalty = float(penalty.get("boundary", 0.0))
        meta = meta or {"tabu_hit": 0, "inverse_hit": 0, "cooldown_hit": 0}
        meta.update(self._meta_base)
        meta["sa_T"] = float(getattr(self, "_sa_T", self.temp_init))
        meta["sa_accept"] = 1 if accept and delta > 0 else 0
        meta["delta_total"] = float(delta)
        line = {
            "iter": int(self._step_id),
            "stage": str(stage),
            "op": str(op_name),
            # ---- for wrapper compatibility ----
            "op_args": op_args,
            "op_args_json": json.dumps({"op": op_name, **op_args}, ensure_ascii=False),
            "assign": list(self.current_solution.assign),
            "seed_id": int(self._seed_id),
            "accepted": 1 if accept else 0,
            "total_scalar": float(new_key),
            "comm_norm": float(new_comm),
            "therm_norm": float(new_therm),
            "pareto_added": 0,
            "duplicate_penalty": duplicate_penalty,
            "boundary_penalty": boundary_penalty,
            "time_ms": int(dt),
            "eval_calls_cum": int(getattr(self._evaluator, "evaluate_calls", 0)),
            "cache_hit_cum": int(getattr(self._evaluator, "cache_hits", 0)) if hasattr(self._evaluator, "cache_hits") else 0,
            "signature": assign_sig,
            "op_signature": op_signature,
            "meta": meta,
        }
        self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._rec_fp.flush()

        self._step_id += 1
        self.current_steps += 1
        return operator

    def run_heuristic(self, heuristic, algorithm_data: dict = {}, record: bool = True, **kwargs):
        t0 = time.time()
        # --- pop internal control args (do NOT pass into heuristic) ---
        _ = kwargs.pop("selection", None)
        add_record_item = kwargs.pop("add_record_item", None)
        meta_in = kwargs.pop("meta", None)

        add_record_item = add_record_item or {}
        stage = add_record_item.get("selection", self._stage)
        meta_full = dict(meta_in or {})
        if isinstance(add_record_item, dict):
            nested = add_record_item.get("meta")
            if isinstance(nested, dict):
                meta_full.update(nested)
            for k, v in add_record_item.items():
                if k == "meta":
                    continue
                meta_full[k] = v
        if not meta_full:
            meta_full = None
        try:
            operator, algorithm_data = heuristic(self.problem_state, algorithm_data, **kwargs)
        except Exception as e:  # noqa: BLE001
            dt = (time.time() - t0) * 1000.0
            metrics = self._eval_assign(list(self.current_solution.assign))
            total_scalar = float(metrics.get("total_scalar", 0.0))
            comm_norm = float(metrics.get("comm_norm", 0.0))
            therm_norm = float(metrics.get("therm_norm", 0.0))
            penalty = metrics.get("penalty", {}) if isinstance(metrics.get("penalty", {}), dict) else {}
            duplicate_penalty = float(penalty.get("duplicate", 0.0))
            boundary_penalty = float(penalty.get("boundary", 0.0))
            assign_sig = _assign_signature(list(self.current_solution.assign))
            op_sig = _op_signature_from_action("error", {})
            meta = {"tabu_hit": 0, "inverse_hit": 0, "cooldown_hit": 0}
            meta.update(self._meta_base)
            if isinstance(meta_full, dict):
                meta.update(meta_full)
            line = {
                "iter": int(self._step_id),
                "stage": str(stage),
                "op": "error",
                # ---- for wrapper compatibility ----
                "op_args": {"error": str(e)},
                "op_args_json": json.dumps({"error": str(e)}, ensure_ascii=False),
                "assign": list(self.current_solution.assign),
                "seed_id": int(self._seed_id),
                "accepted": 0,
                "total_scalar": float(total_scalar),
                "comm_norm": float(comm_norm),
                "therm_norm": float(therm_norm),
                "duplicate_penalty": duplicate_penalty,
                "boundary_penalty": boundary_penalty,
                "pareto_added": 0,
                "time_ms": int(dt),
                "signature": assign_sig,
                "op_signature": op_sig,
                "meta": meta,
            }
            self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._rec_fp.flush()
            self._step_id += 1
            self.current_steps += 1
            return f"[heuristic_error] {e}"

        if not hasattr(operator, "run"):
            dt = (time.time() - t0) * 1000.0
            metrics = self._eval_assign(list(self.current_solution.assign))
            total_scalar = float(metrics.get("total_scalar", 0.0))
            comm_norm = float(metrics.get("comm_norm", 0.0))
            therm_norm = float(metrics.get("therm_norm", 0.0))
            penalty = metrics.get("penalty", {}) if isinstance(metrics.get("penalty", {}), dict) else {}
            duplicate_penalty = float(penalty.get("duplicate", 0.0))
            boundary_penalty = float(penalty.get("boundary", 0.0))
            assign_sig = _assign_signature(list(self.current_solution.assign))
            op_sig = _op_signature_from_action("invalid_operator", {})
            meta = {"tabu_hit": 0, "inverse_hit": 0, "cooldown_hit": 0}
            meta.update(self._meta_base)
            if isinstance(meta_full, dict):
                meta.update(meta_full)
            line = {
                "iter": int(self._step_id),
                "stage": str(stage),
                "op": "invalid_operator",
                # ---- for wrapper compatibility ----
                "op_args": {},
                "op_args_json": json.dumps({}, ensure_ascii=False),
                "assign": list(self.current_solution.assign),
                "seed_id": int(self._seed_id),
                "accepted": 0,
                "total_scalar": float(total_scalar),
                "comm_norm": float(comm_norm),
                "therm_norm": float(therm_norm),
                "duplicate_penalty": duplicate_penalty,
                "boundary_penalty": boundary_penalty,
                "pareto_added": 0,
                "time_ms": int(dt),
                "signature": assign_sig,
                "op_signature": op_sig,
                "meta": meta,
            }
            self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._rec_fp.flush()
            self._step_id += 1
            self.current_steps += 1
            return "[invalid_operator]"

        meta = {"tabu_hit": 0, "inverse_hit": 0, "cooldown_hit": 0}
        meta.update(self._meta_base)
        if isinstance(meta_full, dict):
            meta.update(meta_full)
        dt = (time.time() - t0) * 1000.0
        return self.run_operator(operator, inplace=True, meta=meta, stage=stage, time_ms=dt)

    @property
    def continue_run(self) -> bool:
        if getattr(self, "_max_eval_calls", 0) and self._evaluator is not None:
            if int(getattr(self._evaluator, "evaluator_calls", 0)) >= int(self._max_eval_calls):
                return False
        if self.max_steps is not None:
            return int(self.current_steps) < int(self.max_steps)
        return True

    def dump_result(self):
        if self._rec_fp is not None:
            try:
                self._rec_fp.flush()
                self._rec_fp.close()
            except Exception:
                pass
            self._rec_fp = None
        try:
            from pathlib import Path
            out_dir = Path(self._rec_path).parent if self._rec_path else Path(self.output_dir)
            eval_counter = {
                "eval_calls_total": int(getattr(self._evaluator, "evaluate_calls", 0)),
                "effective_eval_calls_total": int(getattr(self._evaluator, "evaluate_calls", 0)),
                "cache_hits": int(getattr(self._evaluator, "cache_hits", 0))
                if hasattr(self._evaluator, "cache_hits")
                else 0,
                "objective_hash": getattr(self, "_objective_hash", None),
                "seed_id": int(getattr(self, "_seed_id", 0)),
            }
            (out_dir / "eval_counter.json").write_text(
                json.dumps(eval_counter, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        super().dump_result()
