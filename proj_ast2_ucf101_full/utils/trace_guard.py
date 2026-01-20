import os
import json
from datetime import datetime
from typing import Any, Dict


def _utc_iso() -> str:
    # 秒级足够；统一加 Z，避免本地时区混淆
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_trace_file(path: str, header: Dict[str, Any]):
    """
    v5.4: 首行必须是 trace_header，并且包含 payload.signature（完整字段）。
    为兼容旧 reader，保留 type=trace_header 且把 header 展平成顶层字段。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        event = {
            "event_type": "trace_header",
            "type": "trace_header",   # backward-compat
            "time": _utc_iso(),
            "payload": header,
            **header,                 # backward-compat (flatten)
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_trace_event(path: str, event: Dict[str, Any]):
    """
    v5.4: 统一写 event_type + payload。若调用方给的是老格式，尽量自动修复。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ev = dict(event)
    if "event_type" not in ev:
        if "type" in ev:
            ev["event_type"] = ev["type"]
        else:
            ev["event_type"] = "event"

    if "time" not in ev:
        ev["time"] = _utc_iso()

    if "payload" not in ev:
        payload = {k: v for k, v in ev.items() if k not in ("event_type", "type", "time")}
        ev["payload"] = payload

    ev.setdefault("type", ev["event_type"])

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def finalize_trace(path: str, payload: Dict[str, Any], status: str = "ok"):
    """
    v5.4: 最后一行必须是 finalize。
    """
    append_trace_event(
        path,
        {
            "event_type": "finalize",
            "time": _utc_iso(),
            "payload": {"status": status, **(payload or {})},
        },
    )
