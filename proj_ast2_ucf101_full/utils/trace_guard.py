import json
import time
import csv
from pathlib import Path
from typing import Any, Dict, Optional

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
    # 优先 trace_dir/trace.csv，其次 out_dir(trace_dir.parent)/trace.csv
    return [trace_dir / "trace.csv", trace_dir.parent / "trace.csv"]


def _pick_trace_csv_path(trace_dir: Path) -> Path:
    for p in _trace_csv_candidates(trace_dir):
        if p.exists():
            return p
    # 默认写到 out_dir/trace.csv（即 trace_dir.parent）
    return trace_dir.parent / "trace.csv"


def _make_min_row(stage: str, op: str, seed_id: int, iter_id: int, op_args: dict, signature: str) -> dict:
    row = {k: "" for k in TRACE_FIELDS}
    row["iter"] = int(iter_id)
    row["stage"] = stage
    row["op"] = op
    row["op_args_json"] = json.dumps(op_args, ensure_ascii=False)
    row["accepted"] = 1
    row["total_scalar"] = 0.0
    row["comm_norm"] = 0.0
    row["therm_norm"] = 0.0
    row["pareto_added"] = 0
    row["duplicate_penalty"] = 0.0
    row["boundary_penalty"] = 0.0
    row["seed_id"] = int(seed_id)
    row["time_ms"] = 0
    row["objective_scalar"] = 0.0
    row["signature"] = signature  # v5.4: "assign:..."
    row["eval_version"] = "v5.4"
    row["time_unit"] = "ms"
    row["dist_unit"] = "mm"
    # 这些列为可选/统计类，留空或 0 均可
    row["objective_hash"] = stable_hash({"v": "v5.4", "stage": stage, "op": op})
    row["cache_key"] = stable_hash({"sig": signature, "obj": row["objective_hash"]})
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


def _ensure_trace_csv_init(trace_dir: Path, seed_id: int):
    csv_path = _pick_trace_csv_path(trace_dir)
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        w.writeheader()
        # 最小可验 init 行：符合 smoke_trace_schema.py 的约束（signature 必须以 "assign:" 开头）
        w.writerow(_make_min_row("init", "init", seed_id=seed_id, iter_id=0, op_args={}, signature="assign:null"))


def _ensure_trace_csv_finalize(trace_dir: Path, seed_id: int, reason: str, step_hint: int):
    csv_path = _pick_trace_csv_path(trace_dir)
    _ensure_trace_csv_init(trace_dir, seed_id=seed_id)
    last = _read_last_csv_row(csv_path)
    if last and str(last.get("stage", "")) == "finalize" and str(last.get("op", "")) == "finalize":
        return
    # iter 取一个稳定的 hint（避免 -1）
    iter_id = max(0, int(step_hint))
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        w.writerow(_make_min_row("finalize", "finalize", seed_id=seed_id, iter_id=iter_id, op_args={"reason": reason}, signature="assign:null"))


def init_trace_dir(
    trace_dir: Path,
    signature: Dict[str, Any],
    run_meta: Optional[Dict[str, Any]] = None,
    required_signature_keys: Optional[list[str]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
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

    # 1) trace_header.json（结构化元信息）
    _write_json(header_path, {"schema": "v5.4", "signature": signature, "run_meta": run_meta, "ts_ms": int(time.time() * 1000)})

    # 2) manifest.json
    manifest = {"schema": "v5.4", "signature": signature, "run_meta": run_meta, "created_ts_ms": int(time.time() * 1000)}
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(manifest_path, manifest)

    # 3) eval_config_snapshot.yaml（这里只做占位；真正 snapshot 由上层脚本写入即可）
    if not snapshot_path.exists():
        snapshot_path.write_text("# eval_config_snapshot placeholder (v5.4)\n", encoding="utf-8")

    # 4) trace_events.jsonl：确保首条为 trace_header
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")

    run_id = run_meta.get("run_id") or signature.get("run_id") or "unknown"
    if _count_jsonl_lines(events_path) == 0:
        append_trace_event_v54(
            events_path,
            "trace_header",
            payload={"schema": "v5.4", "signature": signature, "run_meta": run_meta},
            run_id=str(run_id),
            step=0,
        )

    # 5) trace.csv：保证至少有 init 行（训练场景也不会因缺 trace.csv 而在收尾崩溃）
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


def append_trace_event_v54(path: Path, event_type: str, payload: dict, run_id: str, step: int):
    path = Path(path)
    flag = path.parent / FINALIZED_FLAG
    if flag.exists() and event_type != "finalize":
        raise RuntimeError(f"trace_events is finalized; refusing to append event_type={event_type}")

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


def update_trace_summary(trace_dir: Path, ok: bool, reason: str, steps_done: int, best_solution_valid: bool):
    trace_dir = Path(trace_dir)
    _write_json(
        trace_dir / "summary.json",
        {
            "schema": "v5.4",
            "ok": bool(ok),
            "reason": str(reason),
            "steps_done": int(steps_done),
            "best_solution_valid": bool(best_solution_valid),
            "ts_ms": int(time.time() * 1000),
        },
    )


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
    _ensure_trace_csv_finalize(trace_dir, seed_id=seed_id, reason=str(summary.get("reason", "done")), step_hint=step_hint)

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
