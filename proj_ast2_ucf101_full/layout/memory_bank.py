"""Structured memory bank (v2-lite) for MPVS.

MemoryBank v1 stores:
  - a coarse condition key (region ids at hot slots)
  - an action sequence (1..K replayable atomic actions)
  - success/fail, cooldown, expiry
  - EWMA gain-per-call proxy (ROI)

Retrieval uses similarity *and* ROI to rank entries, plus light context tags:
  - budget_bucket (coarse budget progress bucket)
  - health_bucket (coarse search-health bucket)

This is intentionally "v2-lite": it improves robustness without heavy tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np


def _hot_slots(traffic_sym: np.ndarray, chip_tdp: Optional[np.ndarray], k: int) -> List[int]:
    t = np.asarray(traffic_sym, dtype=np.float64)
    score = np.sum(t, axis=1)
    if chip_tdp is not None:
        score = score + 1e-6 * np.asarray(chip_tdp, dtype=np.float64)
    idx = np.argsort(-score)[: max(1, min(int(k), score.shape[0]))]
    return [int(x) for x in idx.tolist()]


def build_key(assign: np.ndarray, site_to_region: np.ndarray, hot_slots: List[int]) -> Tuple[int, ...]:
    a = np.asarray(assign, dtype=int)
    reg = np.asarray(site_to_region, dtype=int)[a]
    hs = [int(s) for s in hot_slots if 0 <= int(s) < reg.shape[0]]
    return tuple(int(reg[s]) for s in hs)


def key_similarity(k1: Tuple[int, ...], k2: Tuple[int, ...]) -> float:
    if not k1 or not k2:
        return 0.0
    n = min(len(k1), len(k2))
    if n <= 0:
        return 0.0
    m = 0
    for i in range(n):
        if int(k1[i]) == int(k2[i]):
            m += 1
    return float(m) / float(n)


@dataclass
class MemoryEntry:
    mid: int
    key: Tuple[int, ...]
    actions: List[Dict[str, Any]]
    expire_step: int
    cooldown_until: int = 0
    succ: int = 0
    fail: int = 0
    last_used: int = -1
    last_added: int = -1
    ewma_gain: float = 0.0
    ewma_calls: float = 0.0
    ewma_roi: float = 0.0
    origin_src: str = ""
    # light context tags
    budget_bucket: int = 0
    health_bucket: int = 0


class MemoryBank:
    def __init__(
        self,
        max_size: int = 64,
        ttl_steps: int = 120,
        max_action_len: int = 3,
        hot_slots_k: int = 12,
        ewma_alpha: float = 0.2,
        min_similarity: float = 0.4,
        fail_cooldown: int = 20,
        max_fail: int = 6,
        age_penalty: float = 0.001,
    ) -> None:
        self.max_size = int(max_size)
        self.ttl_steps = int(ttl_steps)
        self.max_action_len = int(max_action_len)
        self.hot_slots_k = int(hot_slots_k)
        self.ewma_alpha = float(ewma_alpha)
        self.min_similarity = float(min_similarity)
        self.fail_cooldown = int(fail_cooldown)
        self.max_fail = int(max_fail)
        self.age_penalty = float(age_penalty)

        self.entries: List[MemoryEntry] = []
        self._next_id = 1
        self.hot_slots: List[int] = []

        # context for scoring (set per step by caller)
        self._ctx_budget_bucket: int = 0
        self._ctx_health_bucket: int = 0

    def init_hot_slots(self, traffic_sym: np.ndarray, chip_tdp: Optional[np.ndarray]) -> None:
        self.hot_slots = _hot_slots(traffic_sym, chip_tdp, self.hot_slots_k)


    @staticmethod
    def _bucket_budget(progress: float) -> int:
        p = float(progress)
        if p <= 0.0:
            return 0
        return int(max(0, min(4, math.floor(p * 5.0))))

    @staticmethod
    def _bucket_health(repeat_ratio: float, blocked_ratio: float) -> int:
        # 0: healthy, 1: warning, 2: distressed
        rr = float(repeat_ratio)
        br = float(blocked_ratio)
        if rr >= 0.75 or br >= 0.50:
            return 2
        if rr >= 0.55 or br >= 0.30:
            return 1
        return 0

    def set_context(self, budget_progress: float, repeat_ratio: float = 0.0, blocked_ratio: float = 0.0) -> None:
        """Set coarse context tags for subsequent add/query scoring."""
        try:
            self._ctx_budget_bucket = int(self._bucket_budget(float(budget_progress)))
            self._ctx_health_bucket = int(self._bucket_health(float(repeat_ratio), float(blocked_ratio)))
        except Exception:
            self._ctx_budget_bucket = 0
            self._ctx_health_bucket = 0

    def tick(self, step: int) -> None:
        step = int(step)
        kept: List[MemoryEntry] = []
        for e in self.entries:
            if int(e.expire_step) < step:
                continue
            if int(e.fail) >= int(self.max_fail) and int(e.succ) <= 0:
                continue
            kept.append(e)
        self.entries = kept

    def _update_roi(self, e: MemoryEntry, gain: float, calls: int) -> None:
        g = float(max(0.0, float(gain)))
        c = float(max(1, int(calls)))
        a = float(self.ewma_alpha)
        e.ewma_gain = (1.0 - a) * float(e.ewma_gain) + a * g
        e.ewma_calls = (1.0 - a) * float(e.ewma_calls) + a * float(c)
        e.ewma_roi = float(e.ewma_gain) / float(max(1e-9, e.ewma_calls))

    def add(
        self,
        assign_before: np.ndarray,
        site_to_region: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp: Optional[np.ndarray],
        actions: List[Dict[str, Any]],
        gain: float,
        step: int,
        origin_src: str = "",
        budget_progress: float = 0.0,
        repeat_ratio: float = 0.0,
        blocked_ratio: float = 0.0,
    ) -> Optional[int]:
        step = int(step)
        if not actions:
            return None
        acts = [dict(a) for a in actions[: max(1, self.max_action_len)]]
        if not self.hot_slots:
            self.init_hot_slots(traffic_sym, chip_tdp)
        key = build_key(assign_before, site_to_region, self.hot_slots)

        mid = int(self._next_id)
        self._next_id += 1
        e = MemoryEntry(
            mid=mid,
            key=key,
            actions=acts,
            expire_step=int(step + self.ttl_steps),
            cooldown_until=0,
            succ=1,
            fail=0,
            last_used=int(step),
            last_added=int(step),
            origin_src=str(origin_src or ""),
            budget_bucket=int(self._bucket_budget(float(budget_progress))),
            health_bucket=int(self._bucket_health(float(repeat_ratio), float(blocked_ratio))),
        )
        self._update_roi(e, gain=float(gain), calls=1)
        self.entries.append(e)

        if len(self.entries) > int(self.max_size):
            self.entries.sort(key=lambda x: (-(float(x.ewma_roi)), -int(x.succ), int(x.fail), -int(x.last_added)))
            self.entries = self.entries[: int(self.max_size)]
        return mid

    def mark_result(self, mid: int, step: int, success: bool, gain: float, calls: int) -> None:
        step = int(step)
        for e in self.entries:
            if int(e.mid) != int(mid):
                continue
            e.last_used = step
            if success:
                e.succ += 1
                self._update_roi(e, gain=float(gain), calls=int(max(1, calls)))
                e.cooldown_until = min(int(e.cooldown_until), step)
            else:
                e.fail += 1
                e.cooldown_until = max(int(e.cooldown_until), step + int(self.fail_cooldown))
            return

    def query(
        self,
        assign: np.ndarray,
        site_to_region: np.ndarray,
        step: int,
        topk: int = 4,
        budget_progress: Optional[float] = None,
        repeat_ratio: float = 0.0,
        blocked_ratio: float = 0.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        step = int(step)
        if not self.entries:
            return []
        if budget_progress is not None:
            self.set_context(float(budget_progress), float(repeat_ratio), float(blocked_ratio))
        key_cur = build_key(assign, site_to_region, self.hot_slots)
        scored: List[Tuple[float, MemoryEntry]] = []
        for e in self.entries:
            if int(e.expire_step) < step:
                continue
            if int(e.cooldown_until) > step:
                continue
            sim = key_similarity(key_cur, e.key)
            if sim < float(self.min_similarity):
                continue
            age = max(0, step - int(e.last_added))
            fail_rate = float(e.fail) / float(max(1, e.succ + e.fail))
            bb = 1.0 + (0.15 if int(e.budget_bucket) == int(self._ctx_budget_bucket) else 0.0)
            hb = 1.0 + (0.10 if int(e.health_bucket) == int(self._ctx_health_bucket) else 0.0)
            score = float(sim) * (float(e.ewma_roi) + 1e-6) * float(bb) * float(hb) - 0.05 * fail_rate - float(self.age_penalty) * float(age)
            scored.append((score, e))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        out: List[Tuple[MemoryEntry, float]] = []
        for s, e in scored[: max(0, int(topk))]:
            out.append((e, float(s)))
        return out

    def snapshot(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in self.entries:
            out.append(
                {
                    "mid": int(e.mid),
                    "key": list(e.key),
                    "n_actions": int(len(e.actions)),
                    "expire_step": int(e.expire_step),
                    "cooldown_until": int(e.cooldown_until),
                    "succ": int(e.succ),
                    "fail": int(e.fail),
                    "last_used": int(e.last_used),
                    "ewma_roi": float(e.ewma_roi),
                    "origin_src": str(e.origin_src),
                }
            )
        return out
