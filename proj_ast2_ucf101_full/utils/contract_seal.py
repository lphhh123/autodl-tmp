from __future__ import annotations

from pathlib import Path
from typing import Any

from .trace_contract_v54 import compute_effective_cfg_digest_v54
from .trace_guard import append_trace_event_v54


def assert_cfg_sealed_or_violate(
    cfg: Any,
    seal_digest: str,
    trace_events_path: Path,
    step: int,
) -> None:
    cur = compute_effective_cfg_digest_v54(cfg)
    if str(cur) != str(seal_digest):
        append_trace_event_v54(
            trace_events_path,
            "contract_violation",
            payload={
                "reason": "cfg_mutated_after_seal",
                "expected_seal_digest": str(seal_digest),
                "actual_seal_digest": str(cur),
            },
            step=int(step),
        )
        raise RuntimeError("[v5.4 P0] cfg mutated after seal_digest")
