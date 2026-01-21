import os
import json
import time
from datetime import datetime
from pathlib import Path
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


def ensure_trace_events(path: Path, payload: dict):
    """
    v5.4 trace_events.jsonl:
      first line MUST be {"event_type":"trace_header","payload":{"signature":{...}, ...},"ts_ms":...}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "signature" not in payload or not isinstance(payload["signature"], dict):
        raise ValueError("trace_header payload must contain dict field 'signature' (v5.4)")

    # required fields check (delegate to builder's hard check if present)
    required = [
        "method_name",
        "config_fingerprint",
        "git_commit_or_version",
        "seed_global",
        "seed_problem",
        "moves_enabled",
        "lookahead_k",
        "bandit_type",
        "policy_switch_mode",
        "cache_enabled",
        "cache_key_schema_version",
        "acc_first_hard_gating_enabled",
        "locked_acc_ref_enabled",
        "acc_ref_source",
        "no_drift_enabled",
        "no_double_scale_enabled",
    ]
    missing = [k for k in required if k not in payload["signature"]]
    if missing:
        raise ValueError(f"trace signature missing required fields: {missing}")

    if path.exists():
        # validate first line
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first:
                obj = json.loads(first)
                if obj.get("event_type") != "trace_header":
                    raise ValueError("trace_events.jsonl first record must be event_type=trace_header")
                hp = obj.get("payload", {})
                if not isinstance(hp, dict) or "signature" not in hp:
                    raise ValueError("trace_header must contain payload.signature")
        except Exception as e:
            raise ValueError(f"invalid existing trace_events file: {path} ({e})")
        return

    append_trace_event_v54(path, "trace_header", payload=payload)


def append_trace_event_v54(path: Path, event_type: str, payload: dict):
    import time
    rec = {
        "event_type": str(event_type),
        "payload": payload if payload is not None else {},
        "ts_ms": int(time.time() * 1000),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def finalize_trace_events(path: Path, payload: dict):
    """
    v5.4 requires:
      payload.reason (steps0/early_stop/error/done)
      payload.steps_done (int)
      payload.best_solution_valid (bool)
    """
    if payload is None:
        payload = {}
    if "reason" not in payload:
        payload["reason"] = "done"
    if "steps_done" not in payload:
        payload["steps_done"] = 0
    if "best_solution_valid" not in payload:
        payload["best_solution_valid"] = False
    append_trace_event_v54(path, "finalize", payload=payload)


REQUIRED_SIGNATURE_KEYS = {
    "method_name",
    "config_fingerprint",
    "git_commit_or_version",
    "seed_global",
    "seed_problem",
    "moves_enabled",
    "lookahead_k",
    "bandit_type",
    "policy_switch_mode",
    "cache_enabled",
    "cache_key_schema_version",
    "acc_first_hard_gating_enabled",
    "locked_acc_ref_enabled",
    "acc_ref_source",
    "no_drift_enabled",
    "no_double_scale_enabled",
}


def validate_required_signature(signature: dict) -> None:
    miss = [k for k in REQUIRED_SIGNATURE_KEYS if k not in signature]
    if miss:
        raise RuntimeError(f"[TRACE] Missing required signature fields: {miss}")
