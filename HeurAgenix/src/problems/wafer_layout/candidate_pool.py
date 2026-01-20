from __future__ import annotations


def inverse_signature(i: int, site_id: int, candidate_id: int, op: str) -> str:
    """
    Keep signature format consistent across Ours/HeurAgenix runs.
    """
    return f"{op}:i={int(i)}:site={int(site_id)}:cand={int(candidate_id)}"
