from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Iterable


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_json_dumps(obj) + "\n")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def update_manifest_json(trace_dir: Path, patch: dict) -> None:
    """
    Update trace_dir/manifest.json in-place (create if missing).
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    p = trace_dir / "manifest.json"
    if p.exists():
        obj = _read_json(p)
    else:
        obj = {"schema": "v5.4", "ts_ms": int(time.time() * 1000)}
    obj.update(patch or {})
    _write_json(p, obj)


# ---------------------------
# v5.4 trace dir API (SPEC_D/E)
# ---------------------------

def init_trace_dir(
    trace_dir: Path,
    *,
    signature: Dict[str, Any],
    run_meta: Optional[Dict[str, Any]] = None,
    required_signature_keys: Optional[Iterable[str]] = None,
) -> None:
    """
    Creates:
      trace/trace_header.json
      trace/trace_events.jsonl   (empty or append-only)
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    if required_signature_keys:
        missing = [k for k in required_signature_keys if k not in signature]
        if missing:
            raise ValueError(f"[trace] missing signature keys: {missing}")

    header_path = trace_dir / "trace_header.json"
    if not header_path.exists():
        header = {
            "ts_ms": _now_ms(),
            "schema": "v5.4",
            "signature": signature,
            "run_meta": run_meta or {},
        }
        _write_json(header_path, header)

    manifest = {
        "schema": "v5.4",
        "ts_ms": _now_ms(),
        "signature": signature,
        "run_meta": run_meta or {},
    }
    _write_json(trace_dir / "manifest.json", manifest)

    summary = {
        "schema": "v5.4",
        "ts_ms": _now_ms(),
        "last_iter": -1,
        "accepted_steps_total": 0,
        "eval_calls_total": 0,
        "wall_time_ms_total": 0,
        "cache_hit_total": 0,
        "cache_miss_total": 0,
        "cache_size": 0,
    }
    _write_json(trace_dir / "summary.json", summary)
    _write_json(trace_dir / "trace_summary.json", summary)

    events_path = trace_dir / "trace_events.jsonl"
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")


def append_trace_event(
    trace_dir: Path,
    *,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    events_path = trace_dir / "trace_events.jsonl"
    if not (trace_dir / "trace_header.json").exists():
        raise RuntimeError("[trace] trace_header.json missing; call init_trace_dir() first")
    _append_jsonl(events_path, {"ts_ms": _now_ms(), "event_type": event_type, "payload": payload})


def update_trace_summary(trace_dir: Path, patch: dict) -> None:
    """
    Update both summary.json and trace_summary.json (back-compat).
    """
    p = trace_dir / "summary.json"
    if p.exists():
        obj = _read_json(p)
    else:
        obj = {"schema": "v5.4", "ts_ms": int(time.time() * 1000)}
    obj.update(patch or {})
    _write_json(p, obj)
    _write_json(trace_dir / "trace_summary.json", obj)


def finalize_trace_dir(
    trace_dir: Path,
) -> None:
    """
    Finalize: verify required files exist.
    """
    trace_csv = trace_dir / "trace.csv"
    if not trace_csv.exists():
        trace_csv = trace_dir.parent / "trace.csv"
    if not trace_csv.exists():
        raise FileNotFoundError(f"[trace_guard] missing required {trace_csv}")
    for fn in ["manifest.json", "summary.json"]:
        p = trace_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"[trace_guard] missing required {p}")


# ---------------------------
# legacy API (backward compat)
# ---------------------------

def ensure_trace_events(path: Path, payload: dict):
    """
    Legacy: path is a jsonl file (out_dir/trace_events.jsonl).
    Keep for compatibility, but do NOT use in new code.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    _append_jsonl(path, {"ts_ms": _now_ms(), "event_type": "trace_header", "payload": payload})


def append_trace_event_v54(path: Path, event_type: str, payload: dict, *, run_id: str, step: int):
    if run_id is None or str(run_id).strip() == "":
        raise ValueError("trace event requires non-empty run_id (v5.4)")
    rec = {
        "ts_ms": _now_ms(),
        "run_id": str(run_id),
        "step": int(step),
        "event_type": str(event_type),
        "payload": payload if payload is not None else {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def finalize_trace_events(path: Path, payload: dict, *, run_id: str, step: int):
    if payload is None:
        payload = {}
    payload.setdefault("reason", "done")
    payload.setdefault("steps_done", 0)
    payload.setdefault("best_solution_valid", False)
    append_trace_event_v54(path, "finalize", payload=payload, run_id=run_id, step=step)
