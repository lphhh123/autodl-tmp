from __future__ import annotations

from pathlib import Path
from typing import Any

from .trace_contract_v54 import compute_effective_cfg_digest_v54
from .trace_guard import append_trace_event_v54


def assert_cfg_sealed_or_violate(
    cfg: Any,
    seal_digest: str | None,
    trace_events_path: Path,
    step: int,
    strict: bool = True,
    *,
    trace_contract: Any = None,
    phase: str | None = None,
    fatal: bool | None = None,
) -> None:
    del trace_contract
    if not seal_digest:
        contract = getattr(cfg, "contract", None)
        seal_digest = getattr(contract, "seal_digest", None)
    seal_digest = str(seal_digest or "")
    cur = compute_effective_cfg_digest_v54(cfg)
    if str(cur) != seal_digest:
        payload = {
            "reason": "cfg_mutated_after_seal",
            "expected_seal_digest": seal_digest,
            "actual_seal_digest": str(cur),
        }
        if phase is not None:
            payload["phase"] = str(phase)
        append_trace_event_v54(
            trace_events_path,
            "contract_violation",
            payload=payload,
            step=int(step),
        )
        should_raise = strict if fatal is None else bool(fatal)
        if should_raise:
            raise RuntimeError("[v5.4 P0] cfg mutated after seal_digest")
