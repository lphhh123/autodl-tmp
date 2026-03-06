"""Verifier engine (v1): lite/full verification + ROI accounting.

Design goals:
  - Under eval-call budgets, verification must be *selective* and *diagnosable*.
  - "Lite" verification should be cheap (0~1 eval call) and screen candidates.
  - "Full" verification should be rare and justified by expected ROI.
  - Provide stable statistics to explain when/why verifier changes decisions.

Used by: layout/detailed_place.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ROIStat:
    ewma_gain: float = 0.0
    ewma_calls: float = 0.0
    ewma_roi: float = 0.0
    n: int = 0


class ROITracker:
    """Tracks per-source ROI (gain per eval-call) with EWMA."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self.by_src: Dict[str, ROIStat] = {}

    def _st(self, src: str) -> ROIStat:
        src = str(src)
        if src not in self.by_src:
            self.by_src[src] = ROIStat()
        return self.by_src[src]

    def update(self, src: str, gain: float, calls: int) -> float:
        """Update ROI for a src. Returns current ewma_roi."""
        st = self._st(src)
        g = float(max(0.0, float(gain)))
        c = float(max(0, int(calls)))
        if c <= 0.0:
            # 0-call items (e.g., cand.est) do not inform ROI.
            return float(st.ewma_roi)
        a = float(self.alpha)
        st.n += 1
        st.ewma_gain = (1.0 - a) * float(st.ewma_gain) + a * g
        st.ewma_calls = (1.0 - a) * float(st.ewma_calls) + a * c
        st.ewma_roi = float(st.ewma_gain) / float(max(1e-9, st.ewma_calls))
        return float(st.ewma_roi)

    def roi(self, src: str, default: float = 0.0) -> float:
        st = self.by_src.get(str(src))
        return float(st.ewma_roi) if st is not None else float(default)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, st in self.by_src.items():
            out[str(k)] = {
                "n": int(st.n),
                "ewma_gain": float(st.ewma_gain),
                "ewma_calls": float(st.ewma_calls),
                "ewma_roi": float(st.ewma_roi),
            }
        return out


def compute_gain(cur_total: float, verified_total: float) -> float:
    """Positive gain means improvement (lower is better)."""
    return float(max(0.0, float(cur_total) - float(verified_total)))
