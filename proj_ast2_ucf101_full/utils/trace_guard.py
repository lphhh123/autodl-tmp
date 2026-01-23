import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .trace_schema import TRACE_FIELDS
from .stable_hash import stable_hash

FINALIZED_FLAG = "finalized.flag"


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _trace_csv_candidates(trace_dir: Path) -> list[Path]:
    """
    v5.4 contract: trace.csv should live at run out_dir root (trace_dir.parent),
    while trace_events.jsonl/summary.json live under out_dir/trace/.
    """
    return [
        trace_dir.parent / "trace.csv",
        trace_dir / "trace.csv",
    ]


def _pick_trace_csv_path(trace_dir: Path) -> Path:
    for p in _trace_csv_candidates(trace_dir):
        if p.exists():
            return p
    # 默认写到 out_dir/trace.csv（即 trace_dir.parent）
    return trace_dir.parent / "trace.csv"


def _make_min_row(
    stage: str,
    op: str,
    seed_id: int,
    iter_id: int,
    op_args: Dict[str, Any],
    signature: str,
    total_scalar: float = 0.0,
    comm_norm: float = 0.0,
    therm_norm: float = 0.0,
    objective_hash: str = "",
    time_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Minimal trace row that will NOT break csv writing and downstream int/float casts.
    Must only use keys in TRACE_FIELDS.
    """
    row = {k: "" for k in TRACE_FIELDS}
    row["iter"] = int(iter_id)
    row["stage"] = str(stage)
    row["op"] = str(op)
    row["op_args_json"] = json.dumps(op_args or {}, ensure_ascii=False)
    row["accepted"] = 1 if stage == "init" else 0
    row["seed_id"] = int(seed_id)
    row["signature"] = str(signature)

    row["total_scalar"] = float(total_scalar)
    row["comm_norm"] = float(comm_norm)
    row["therm_norm"] = float(therm_norm)

    row["delta_total"] = 0.0
    row["delta_comm"] = 0.0
    row["delta_therm"] = 0.0

    row["pareto_added"] = 0
    row["duplicate_penalty"] = 0.0
    row["boundary_penalty"] = 0.0

    row["tabu_hit"] = 0
    row["tabu_accept_override"] = 0
    row["repeat_signature"] = 0
    row["undo_signature"] = 0
    row["policy_switch"] = ""
    row["cache_hit"] = 0
    row["cache_enabled"] = 0
    row["eval_calls_cum"] = 0
    row["cache_hit_cum"] = 0
    row["accepted_steps_cum"] = 0
    row["score_lookahead"] = 0.0
    row["score_tts"] = 0.0
    row["score_llm"] = 0.0
    row["selection_id"] = ""
    row["step_id"] = ""
    row["heuristic_name"] = ""

    row["time_ms"] = float(time_ms)
    row["time_s"] = float(time_ms) / 1000.0

    row["objective_hash"] = str(objective_hash)
    row["cache_key"] = stable_hash({"signature": signature, "seed_id": int(seed_id), "objective_hash": str(objective_hash)})

    return row


def _read_last_csv_row(csv_path: Path) -> Optional[dict]:
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            last = r
    return last


def _ensure_trace_csv_init(trace_dir: Path, seed_id: int) -> Path:
    trace_csv = _pick_trace_csv_path(trace_dir)
    trace_csv.parent.mkdir(parents=True, exist_ok=True)
    if not trace_csv.exists():
        with trace_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, restval="", extrasaction="ignore")
            w.writeheader()
            w.writerow(
                _make_min_row(
                    stage="init",
                    op="init",
                    seed_id=seed_id,
                    iter_id=-1,
                    op_args={"op": "init"},
                    signature="assign:unknown",
                    total_scalar=0.0,
                    comm_norm=0.0,
                    therm_norm=0.0,
                    objective_hash="",
                    time_ms=0.0,
                )
            )
    return trace_csv


def _ensure_trace_csv_finalize(trace_dir: Path) -> None:
    trace_csv = _pick_trace_csv_path(trace_dir)
    if not trace_csv.exists():
        _ensure_trace_csv_init(trace_dir, seed_id=0)

    last = _read_last_csv_row(trace_csv)
    if last and last.get("stage") == "finalize":
        return

    iter_id = int(last.get("iter", 0)) + 1 if last else 0
    seed_id = int(last.get("seed_id", 0)) if last else 0
    objective_hash = str(last.get("objective_hash", "")) if last else ""

    with trace_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, restval="", extrasaction="ignore")
        w.writerow(
            _make_min_row(
                stage="finalize",
                op="finalize",
                seed_id=seed_id,
                iter_id=iter_id,
                op_args={"op": "finalize"},
                signature="assign:unknown",
                total_scalar=float(last.get("total_scalar", 0.0)) if last else 0.0,
                comm_norm=float(last.get("comm_norm", 0.0)) if last else 0.0,
                therm_norm=float(last.get("therm_norm", 0.0)) if last else 0.0,
                objective_hash=objective_hash,
                time_ms=float(last.get("time_ms", 0.0)) if last else 0.0,
            )
        )


def init_trace_dir(
    trace_dir: Path,
    signature: Dict[str, Any],
    run_meta: Optional[Dict[str, Any]] = None,
    required_signature_keys: Optional[list[str]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Create/initialize a trace directory with v5.4 required artifacts.

    SPEC_E requirements:
    - trace_header.json must include resolved effective config (after defaults/overrides).
    - eval_config_snapshot.yaml must contain the resolved config.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 新 run：清掉 finalized 标记
    flag = trace_dir / FINALIZED_FLAG
    if flag.exists():
        try:
            flag.unlink()
        except Exception:
            pass

    if required_signature_keys:
        missing = [k for k in required_signature_keys if k not in signature]
        if missing:
            raise KeyError(f"trace signature missing keys: {missing}")

    run_meta = {} if run_meta is None else dict(run_meta)

    header_path = trace_dir / "trace_header.json"
    manifest_path = trace_dir / "manifest.json"
    snapshot_path = trace_dir / "eval_config_snapshot.yaml"
    events_path = trace_dir / "trace_events.jsonl"

    # 0) Write resolved config snapshot (YAML)
    resolved_cfg_obj = None
    if resolved_config is not None:
        resolved_cfg_obj = dict(resolved_config)
        with snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_cfg_obj, f, sort_keys=False, allow_unicode=True)
    else:
        if not snapshot_path.exists():
            snapshot_path.write_text("# eval_config_snapshot placeholder (v5.4)\n", encoding="utf-8")

    # 1) trace_header.json（结构化元信息）
    header_payload = {
        "schema": "v5.4",
        "signature": signature,
        "run_meta": run_meta,
        "resolved_config": resolved_cfg_obj,
        "ts_ms": int(time.time() * 1000),
    }
    _write_json(header_path, header_payload)

    # 2) manifest.json
    manifest = {
        "schema": "v5.4",
        "signature": signature,
        "run_meta": run_meta,
        "resolved_config_path": str(snapshot_path),
        "created_ts_ms": int(time.time() * 1000),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(manifest_path, manifest)

    # 3) trace_events.jsonl：确保首条为 trace_header
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")

    run_id = run_meta.get("run_id") or signature.get("run_id") or "unknown"
    if _count_jsonl_lines(events_path) == 0:
        append_trace_event_v54(
            events_path,
            "trace_header",
            payload={
                "schema": "v5.4",
                "signature": signature,
                "run_meta": run_meta,
                "resolved_config": resolved_cfg_obj,
            },
            run_id=str(run_id),
            step=0,
        )

    # 4) trace.csv：保证至少有 init 行（训练场景也不会因缺 trace.csv 而在收尾崩溃）
    seed_id = int(run_meta.get("seed_id", 0) or 0)
    _ensure_trace_csv_init(trace_dir, seed_id=seed_id)

    return {
        "trace_dir": trace_dir,
        "trace_header": header_path,
        "manifest": manifest_path,
        "eval_config_snapshot": snapshot_path,
        "trace_events": events_path,
        "trace_csv": _pick_trace_csv_path(trace_dir),
    }


def init_trace_dir_v54(
    base_dir: Path,
    run_id: str,
    cfg: Any,
    signature: Dict[str, Any],
    signature_v54: Dict[str, Any],
    required_signature_fields: Optional[list[str]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    global _DEFAULT_RUN_ID
    _DEFAULT_RUN_ID = str(run_id)
    trace_dir = Path(base_dir) / str(run_id)
    resolved_config = None
    if cfg is not None:
        try:
            import omegaconf

            resolved_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            resolved_config = None
    meta = init_trace_dir(
        trace_dir=trace_dir,
        signature=signature_v54 or signature,
        run_meta=run_meta,
        required_signature_keys=required_signature_fields,
        extra_manifest=extra_manifest,
        resolved_config=resolved_config,
    )
    snapshot_path = Path(meta["eval_config_snapshot"])
    if cfg is not None:
        try:
            import omegaconf

            snapshot_path.write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
        except Exception:
            snapshot_path.write_text(str(cfg), encoding="utf-8")
    return meta


_DEFAULT_RUN_ID: Optional[str] = None


def _require_keys(event_type: str, payload: dict, keys: list) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"[SPEC_E v5.4] trace event '{event_type}' missing keys: {missing}")


def append_trace_event_v54(
    path: Path,
    event_type: str,
    payload: Optional[dict] = None,
    run_id: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    outer_iter: Optional[int] = None,
):
    if payload is None:
        payload = {}

    # hard forbid legacy / ambiguous event_type aliases
    if event_type in {"gating_decision", "gatingDecision", "gate_decision"}:
        raise ValueError(
            "[SPEC_E v5.4] Illegal trace event_type alias. Use event_type='gating' with payload.gate in "
            "{'allow_hw','reject_hw'}."
        )

    # schema enforcement for contract-critical events
    if event_type == "gating":
        _require_keys(
            event_type,
            payload,
            ["acc_ref", "acc_now", "acc_drop", "acc_drop_max", "gate", "hw_loss_raw", "hw_loss_used", "total_loss"],
        )
        if payload["gate"] not in ("allow_hw", "reject_hw"):
            raise ValueError(
                f"[SPEC_E v5.4] gating.gate must be 'allow_hw' or 'reject_hw', got: {payload['gate']}"
            )
    elif event_type == "proxy_sanitize":
        _require_keys(event_type, payload, ["metric", "raw_value", "used_value", "penalty_added"])
    elif event_type == "ref_update":
        _require_keys(event_type, payload, ["ref_name", "old_value", "new_value", "reason"])

    path = Path(path)
    flag = path.parent / FINALIZED_FLAG
    if flag.exists() and event_type != "finalize":
        raise RuntimeError(f"trace_events is finalized; refusing to append event_type={event_type}")

    if run_id is None:
        run_id = _DEFAULT_RUN_ID or "unknown"
    if step is None:
        if outer_iter is not None:
            step = int(outer_iter)
        elif epoch is not None:
            step = int(epoch)
        else:
            step = 0

    _append_jsonl(
        path,
        {
            "ts_ms": int(time.time() * 1000),
            "run_id": str(run_id),
            "step": int(step),
            "event_type": str(event_type),
            "payload": payload,
        },
    )


def finalize_trace_events(path: Path, payload: dict, run_id: str, step: int):
    append_trace_event_v54(path, "finalize", payload=payload, run_id=run_id, step=step)


def update_trace_summary(
    trace_dir: Path,
    ok: Any,
    reason: Optional[str] = None,
    steps_done: Optional[int] = None,
    best_solution_valid: Optional[bool] = None,
) -> None:
    """
    v5.4 compat:
    - Accept both legacy signature:
        update_trace_summary(trace_dir, ok, reason, steps_done, best_solution_valid)
    - And new call style used across this repo:
        update_trace_summary(trace_dir, {"reason":..., "steps_done":..., "best_solution_valid":..., ...})
    """
    trace_dir = Path(trace_dir)

    extra = {}
    if isinstance(ok, dict) and reason is None and steps_done is None and best_solution_valid is None:
        payload = ok
        reason = str(payload.get("reason", "unknown"))
        steps_done = int(payload.get("steps_done", 0) or 0)
        best_solution_valid = bool(payload.get("best_solution_valid", False))
        ok_val = payload.get("ok", None)
        if ok_val is None:
            ok_val = reason in ("done", "steps0", "ok", "success")
        ok = bool(ok_val)

        extra = {
            k: v
            for k, v in payload.items()
            if k not in ("ok", "reason", "steps_done", "best_solution_valid")
        }
    else:
        ok = bool(ok)
        reason = str(reason if reason is not None else "unknown")
        steps_done = int(steps_done if steps_done is not None else 0)
        best_solution_valid = bool(best_solution_valid if best_solution_valid is not None else False)

    trace_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "v5.4",
        "ok": ok,
        "reason": reason,
        "steps_done": steps_done,
        "best_solution_valid": best_solution_valid,
        "ts_ms": int(time.time() * 1000),
    }
    if extra:
        out.update(extra)

    _write_json(trace_dir / "summary.json", out)


def ensure_trace_events(trace_dir: Path) -> Path:
    """
    兼容旧逻辑：保证 trace_events.jsonl 至少存在；真正的 trace_header/finalize 由 init/finalize 管控。
    """
    trace_dir = Path(trace_dir)
    p = trace_dir / "trace_events.jsonl"
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def finalize_trace_dir(trace_dir: Path):
    """
    v5.4 合同收尾：
    - 确保 summary.json 存在（否则写一个最小的）
    - 确保 trace.csv 最后一行是 finalize（否则自动补一行）
    - 追加 trace_events.jsonl 的 finalize 事件，并写 finalized.flag，禁止后续再 append
    - 最后做 required files 校验
    """
    trace_dir = Path(trace_dir)
    required = ["trace_header.json", "manifest.json", "eval_config_snapshot.yaml", "trace_events.jsonl"]

    # 0) summary.json 兜底
    summary_path = trace_dir / "summary.json"
    if not summary_path.exists():
        update_trace_summary(trace_dir, ok=False, reason="missing_summary_autofill", steps_done=0, best_solution_valid=False)

    summary = _read_json(summary_path, default={})
    manifest = _read_json(trace_dir / "manifest.json", default={})
    run_meta = manifest.get("run_meta", {}) or {}
    signature = manifest.get("signature", {}) or {}

    run_id = run_meta.get("run_id") or signature.get("run_id") or "unknown"
    seed_id = int(run_meta.get("seed_id", 0) or 0)
    step_hint = max(0, int(summary.get("steps_done", 0) or 0))

    # 1) trace.csv finalize 行兜底
    _ensure_trace_csv_finalize(trace_dir)

    # 2) finalize 事件：只允许写一次，并且必须是最后一条
    flag = trace_dir / FINALIZED_FLAG
    events_path = trace_dir / "trace_events.jsonl"
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")

    if not flag.exists():
        finalize_trace_events(
            events_path,
            payload={
                "schema": "v5.4",
                "ok": bool(summary.get("ok", False)),
                "reason": str(summary.get("reason", "done")),
                "steps_done": int(summary.get("steps_done", 0) or 0),
                "best_solution_valid": bool(summary.get("best_solution_valid", False)),
            },
            run_id=str(run_id),
            step=int(step_hint),
        )
        flag.write_text(f"finalized_ts_ms={int(time.time() * 1000)}\n", encoding="utf-8")

    # 3) required files 校验
    for name in required:
        p = trace_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing required trace file: {p}")

    # trace.csv 也必须存在（两处之一）
    csv_ok = any(p.exists() for p in _trace_csv_candidates(trace_dir))
    if not csv_ok:
        raise FileNotFoundError(f"Missing required trace.csv at either {trace_dir/'trace.csv'} or {trace_dir.parent/'trace.csv'}")
