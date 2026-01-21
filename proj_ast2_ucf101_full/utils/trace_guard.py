import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional


def ensure_trace_file(path: str, header: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "trace_header",
                        "time": datetime.utcnow().isoformat(),
                        **header,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_trace_event(path: str, event: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ============================
# v5.4 Trace Events (JSONL)
# ============================


def _now_ts_ms() -> int:
    return int(time.time() * 1000)


def ensure_trace_events(path: str, run_id: str, payload: Dict[str, Any], step: int = -1) -> None:
    """
    Create trace_events.jsonl with a mandatory trace_header event.
    v5.4 schema: {ts_ms, run_id, step, event_type, payload}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    header = {
        "ts_ms": _now_ts_ms(),
        "run_id": str(run_id),
        "step": int(step),
        "event_type": "trace_header",
        "payload": dict(payload) if isinstance(payload, dict) else {"payload": str(payload)},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def append_trace_event_v54(
    path: str,
    run_id: str,
    step: int,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    ev = {
        "ts_ms": _now_ts_ms(),
        "run_id": str(run_id),
        "step": int(step),
        "event_type": str(event_type),
        "payload": dict(payload) if isinstance(payload, dict) else {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def finalize_trace_events(
    path: str,
    run_id: str,
    step: Optional[int] = None,
    reason: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    reason_value = str(reason or payload_dict.get("reason", "unknown"))
    step_value = int(step) if step is not None else int(payload_dict.get("steps_done", payload_dict.get("step", -1)))
    payload_dict["reason"] = reason_value
    ev = {
        "ts_ms": _now_ts_ms(),
        "run_id": str(run_id),
        "step": step_value,
        "event_type": "finalize",
        "payload": payload_dict,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


REQUIRED_SIGNATURE_KEYS = [
    # Ours-B2+
    "moves_enabled",
    "lookahead_k",
    "bandit_type",
    "policy_switch_mode",
    "cache_enabled",
    "cache_key_schema_version",
    # Acc-first / locked / stability
    "acc_first_hard_gating_enabled",
    "locked_acc_ref_enabled",
    "acc_ref_source",
    "no_drift_enabled",
    "no_double_scale_enabled",
    # Repro
    "seed_global",
    "seed_problem",
    "config_fingerprint",
    "git_commit_or_version",
]


def validate_required_signature(signature: dict) -> None:
    miss = [k for k in REQUIRED_SIGNATURE_KEYS if k not in signature]
    if miss:
        raise RuntimeError(f"[TRACE] Missing required signature fields: {miss}")
