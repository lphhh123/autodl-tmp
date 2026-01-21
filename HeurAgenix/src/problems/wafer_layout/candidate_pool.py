from __future__ import annotations


def inverse_signature(i: int, site_id: int, candidate_id: int, op: str) -> str:
    """
    Operator/action signature (op-level), used for tabu/inverse checks.
    NOTE: recordings.jsonl field `signature` is assign-level signature ("assign:...").
    This op signature should be stored in `op_signature` when needed.
    """
    return f"{op}:i={int(i)}:site={int(site_id)}:cand={int(candidate_id)}"
