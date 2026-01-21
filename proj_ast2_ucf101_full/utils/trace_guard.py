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


def finalize_trace_dir(
    trace_dir: Path,
    *,
    summary_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates trace/trace_summary.json and appends a finalize event.
    """
    events_path = trace_dir / "trace_events.jsonl"
    header_path = trace_dir / "trace_header.json"
    if not header_path.exists():
        raise RuntimeError("[trace] trace_header.json missing; call init_trace_dir() first")

    n_events = _count_jsonl_lines(events_path)
    summary = {
        "ts_ms": _now_ms(),
        "schema": "v5.4",
        "n_events": n_events,
    }
    if summary_extra:
        summary.update(summary_extra)

    append_trace_event(trace_dir, event_type="finalize", payload={"summary": summary})
    _write_json(trace_dir / "trace_summary.json", summary)
    return summary


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


def append_trace_event_v54(path: Path, event_type: str, payload: dict):
    _append_jsonl(path, {"ts_ms": _now_ms(), "event_type": event_type, "payload": payload})


def finalize_trace_events(path: Path, payload: dict):
    _append_jsonl(path, {"ts_ms": _now_ms(), "event_type": "finalize", "payload": payload})
