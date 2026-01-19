from __future__ import annotations

from typing import Iterable, List


def signature_from_assign(assign: Iterable[int]) -> str:
    arr = list(map(int, assign))
    return "assign:" + ",".join(map(str, arr))


# backward-compat
def signature_for_assign(assign: List[int]) -> str:
    return signature_from_assign(assign)
