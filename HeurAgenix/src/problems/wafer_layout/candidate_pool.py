from __future__ import annotations

from typing import Iterable


def signature_from_assign(assign: Iterable[int]) -> str:
    """
    v5.4 canonical signature: assignment signature
    Format: "assign:0,1,2,..."
    """
    try:
        arr = [int(x) for x in list(assign)]
    except Exception:
        arr = []
    return "assign:" + ",".join(str(x) for x in arr)


def op_signature(op: str, i: int, site_id: int, candidate_id: int) -> str:
    """
    Optional debug signature for an operator action (NOT used as v5.4 trace signature).
    """
    return f"{op}:i={int(i)}:site={int(site_id)}:cand={int(candidate_id)}"
