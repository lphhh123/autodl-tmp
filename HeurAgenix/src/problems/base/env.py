"""Base environment for HeurAgenix problems."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_env_seed_id(env) -> int:
    if hasattr(env, "_seed_id"):
        return _safe_int(getattr(env, "_seed_id"), 0)
    if hasattr(env, "seed_id"):
        return _safe_int(getattr(env, "seed_id"), 0)
    if hasattr(env, "seed"):
        return _safe_int(getattr(env, "seed"), 0)
    return 0


def _signature_from_assign(assign) -> str:
    """
    Canonical signature for an assignment vector.
    Must match proj_ast2_ucf101_full/layout/candidate_pool.py: signature_from_assign
    Format: "assign:" + comma-joined ints
    """
    try:
        import numpy as np

        a = np.asarray(assign, dtype=int).reshape(-1)
        return "assign:" + ",".join(str(int(x)) for x in a.tolist())
    except Exception:
        return "assign:unknown"


class BaseEnv:
    def __init__(self, data_path: str, problem: Optional[str] = None):
        self.data_path = data_path
        self.data_name = data_path
        self.problem = problem or ""
        # 1) load instance first
        self.instance_data = self.load_data(data_path)

        # 2) allow subclass to initialize attributes needed by init_solution()
        # (e.g., problem size, RNG/seed, derived constants)
        setup_fn = getattr(self, "setup_from_instance_data", None)
        if callable(setup_fn):
            setup_fn(self.instance_data)

        # 3) now it is safe to build initial solution
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings: List[Dict[str, Any]] = []
        self.step = 0
        self.algorithm_data: Dict[str, Any] = {}
        self.output_dir: Optional[str] = None
        self.current_steps = 0
        self.max_steps = None
        self.seed = 0
        self._rec_fp = None
        self._rec_path = None

    def load_data(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def init_solution(self):
        raise NotImplementedError

    def get_key_value(self, solution) -> float:
        raise NotImplementedError

    def get_problem_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def summarize_env(self) -> Dict[str, Any]:
        if hasattr(self, "get_problem_state"):
            try:
                return dict(self.get_problem_state())
            except Exception:  # noqa: BLE001
                return {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}
        return {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}

    def reset(self, output_dir: Optional[str] = None) -> None:
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings = []
        self.step = 0
        self.construction_steps = 0
        self.current_steps = 0
        self.algorithm_data = {}
        self.output_dir = output_dir
        if self._rec_fp is not None:
            try:
                self._rec_fp.close()
            except Exception:
                pass
        self._rec_fp = None
        self._rec_path = None
        self._ensure_rec_fp()
        try:
            init_action = {"op": "init"}
            sig = None
            try:
                if hasattr(self.solution, "assign") and isinstance(getattr(self.solution, "assign"), (list, tuple)):
                    sig = _signature_from_assign(getattr(self.solution, "assign"))
            except Exception:
                sig = None

            seed_id = _get_env_seed_id(self)
            key_value = self.get_key_value(self.solution)

            # v5.4 unified schema init record (so schema checks do not fail on line-1)
            init_record = {
                "iter": 0,
                "stage": "init",
                "op": "init",
                "op_args": init_action,
                "op_args_json": "{}",
                "accepted": 1,

                # unified scalar fields (default 0 for init)
                "total_scalar": float(key_value) if key_value is not None else 0.0,
                "comm_norm": 0.0,
                "therm_norm": 0.0,
                "pareto_added": 0,
                "duplicate_penalty": 0.0,
                "boundary_penalty": 0.0,

                # required
                "seed_id": int(seed_id),
                "time_ms": 0,

                # signature must exist even at init
                "signature": sig if sig is not None else "init",

                # keep legacy fields (harmless)
                "key_value": float(key_value) if key_value is not None else None,
                "solution_value": float(self.solution.get_solution_value())
                if hasattr(self.solution, "get_solution_value")
                else None,
                "timestamp": float(time.time()),
            }
            self._write_record(init_record)
        except Exception:
            pass

    def _ensure_rec_fp(self):
        if not self.output_dir:
            return None
        if self._rec_fp is not None:
            return self._rec_fp
        os.makedirs(self.output_dir, exist_ok=True)
        self._rec_path = os.path.join(self.output_dir, "recordings.jsonl")
        self._rec_fp = open(self._rec_path, "a", encoding="utf-8")
        return self._rec_fp

    def _write_record(self, line: Dict[str, Any]) -> None:
        fp = self._ensure_rec_fp()
        if fp is None:
            return
        fp.write(json.dumps(line, ensure_ascii=False) + "\n")
        fp.flush()

    def run_operator(
        self,
        operator,
        inplace: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        stage: str = "heuragenix",
        time_ms: Optional[float] = None,
        heuristic_name: Optional[str] = None,
        add_record_item: Optional[Dict[str, Any]] = None,
        new_solution=None,
    ) -> bool:
        """Apply operator to current_solution and advance one construction step."""
        from .components import BaseOperator

        _ = (inplace, meta, stage, time_ms, heuristic_name, add_record_item, new_solution)

        if not isinstance(operator, BaseOperator):
            return False
        self.current_solution = operator.run(self.current_solution)
        self.problem_state = self.get_problem_state()
        if getattr(self, "construction_steps", None) is None:
            self.construction_steps = 0
        self.construction_steps += 1
        self.current_steps += 1
        seed_id = _get_env_seed_id(self)
        record = {
            "iter": int(self.current_steps),
            "stage": stage,
            "op": getattr(operator, "name", operator.__class__.__name__),
            "heuristic_name": heuristic_name,
            "time_ms": float(time_ms) if time_ms is not None else None,
            "meta": meta or {},
            "seed_id": int(seed_id),
        }
        action = operator.to_action() if hasattr(operator, "to_action") else {"op": type(operator).__name__}
        record["op_args"] = action
        record["op_args_json"] = json.dumps(action, ensure_ascii=False)
        if add_record_item:
            record.update(add_record_item)
        if not record.get("_skip_record"):
            sig = None
            try:
                if hasattr(self.current_solution, "assign"):
                    a = getattr(self.current_solution, "assign")
                    if isinstance(a, (list, tuple)):
                        sig = _signature_from_assign(a)
            except Exception:
                sig = None
            if sig is not None:
                record["signature"] = sig
            try:
                record["key_value"] = float(self.get_key_value(self.current_solution))
            except Exception:
                record["key_value"] = None
            record["timestamp"] = float(time.time())
            self._write_record(record)
        return True

    def run_heuristic(self, heuristic, algorithm_data: dict = {}, record: bool = True, **kwargs):
        """
        IMPORTANT:
          - selection/add_record_item/meta are INTERNAL control params; do NOT pass into heuristic().
          - add_record_item must be merged into recordings by run_operator.
        """
        selection = kwargs.pop("selection", None)
        add_record_item = kwargs.pop("add_record_item", None)
        meta = kwargs.pop("meta", None)

        problem_state = self.problem_state
        if algorithm_data is None:
            algorithm_data = {}

        operator, algorithm_data = heuristic(problem_state, algorithm_data=algorithm_data, **kwargs)

        if not record:
            add_record_item = {"_skip_record": True}
        else:
            merged = {}
            if isinstance(add_record_item, dict):
                merged.update(add_record_item)
            if selection is not None:
                merged.setdefault("selection", selection)
            if meta is not None:
                merged.setdefault("meta", {})
                if isinstance(merged["meta"], dict) and isinstance(meta, dict):
                    merged["meta"].update(meta)
            add_record_item = merged if merged else None

        try:
            self.run_operator(operator, add_record_item=add_record_item)
        except TypeError:
            self.run_operator(operator)

        return operator, algorithm_data

    def dump_result(self) -> None:
        return None

    def validate_solution(self, solution=None) -> bool:
        """
        Default validator.
        Most built-in problems implement `validation_solution(solution=None)`.
        wafer_layout implements `validate_solution(solution)`.
        """
        vs = getattr(self, "validation_solution", None)
        if callable(vs):
            try:
                return bool(vs(solution))
            except TypeError:
                return bool(vs())
        return True

    @property
    def is_valid_solution(self) -> bool:
        """
        Unified validity flag (bool property).
        It will call `validate_solution(...)` (possibly overridden), and fall back to True.
        """
        try:
            return bool(self.validate_solution(getattr(self, "current_solution", None)))
        except TypeError:
            return bool(self.validate_solution())
        except Exception:
            return True

    @property
    def continue_run(self) -> bool:
        """
        Budget-first stopping rule:
        - If max_steps is set: stop ONLY by current_steps >= max_steps (SPEC budget semantics).
        - If max_steps is None: allow early stop by is_complete_solution.
        """
        if self.max_steps is not None:
            return int(self.current_steps) < int(self.max_steps)

        done = getattr(self, "is_complete_solution", None)
        if done is None:
            return True
        return not bool(done() if callable(done) else done)
