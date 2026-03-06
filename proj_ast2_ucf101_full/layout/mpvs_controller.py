"""MPVS unified multi-component controller (v1).

This controller gates expensive enhancement sources under eval-call budgets:
  - MacroEngine (macro)
  - MemoryBank (mem)
  - LLM proposer (llm)

Key design goals:
  - deterministic + lightweight (no evaluator calls)
  - ROI-aware (gain/call) + fail-streak cooldown (suppresses harmful sources)
  - global memory gate to avoid noisy memory degrading search health
  - exposes snapshot for trace_meta diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class CompState:
    name: str
    enabled: bool = True
    last_fire_step: int = -10**9
    cooldown_until: int = 0
    fail_streak: int = 0
    fired: int = 0
    allow_last: bool = False
    deny_last_reason: str = ""


class MPVSController:
    def __init__(self, cfg: Dict[str, Any], instance_tag: str = "") -> None:
        self.cfg = cfg or {}
        self.instance_tag = str(instance_tag or "")
        self.states: Dict[str, CompState] = {
            "macro": CompState("macro"),
            "mem": CompState("mem"),
            "llm": CompState("llm"),
        }

        memg = (self.cfg.get("mem_global") or {}) if isinstance(self.cfg, dict) else {}
        self.mem_window = int(memg.get("window", 40))
        self.mem_fail_rate_hi = float(memg.get("fail_rate_hi", 0.75))
        self.mem_roi_floor = float(memg.get("roi_floor", 0.0))
        self.mem_global_cooldown = int(memg.get("cooldown_steps", 30))
        self.mem_global_until = 0
        self._mem_hist: list[int] = []

    def _get_cfg(self, comp: str) -> Dict[str, Any]:
        c = self.cfg.get(str(comp), {}) if isinstance(self.cfg, dict) else {}
        return c or {}

    def tick(self, step: int) -> None:
        step = int(step)
        for st in self.states.values():
            st.allow_last = False
            st.deny_last_reason = ""
        if self.mem_window > 0 and len(self._mem_hist) > 4 * self.mem_window:
            self._mem_hist = self._mem_hist[-4 * self.mem_window :]
        if self.mem_global_until < 0:
            self.mem_global_until = 0

    def allow(self, comp: str, step: int, stagn: int, distress: float, repeat_ratio: float, roi: float) -> Tuple[bool, str]:
        step = int(step)
        comp = str(comp)
        st = self.states.get(comp)
        if st is None:
            return False, "unknown_comp"
        cfg = self._get_cfg(comp)

        st.enabled = bool(cfg.get("enabled", True))
        if not st.enabled:
            st.allow_last = False
            st.deny_last_reason = "disabled"
            return False, "disabled"

        if comp == "mem" and step < int(self.mem_global_until):
            st.allow_last = False
            st.deny_last_reason = "mem_global_cooldown"
            return False, "mem_global_cooldown"

        if step < int(st.cooldown_until):
            st.allow_last = False
            st.deny_last_reason = "cooldown"
            return False, "cooldown"

        min_interval = int(cfg.get("min_interval_steps", 0))
        if step - int(st.last_fire_step) < max(0, min_interval):
            st.allow_last = False
            st.deny_last_reason = "min_interval"
            return False, "min_interval"

        stagn_ge = int(cfg.get("enable_when_stagnation_ge", 0))
        if int(stagn) < int(stagn_ge):
            st.allow_last = False
            st.deny_last_reason = "stagnation"
            return False, "stagnation"

        if comp == "llm":
            dist_ge = float(cfg.get("enable_when_distress_ge", 0.7))
            rep_ge = float(cfg.get("enable_when_repeat_ratio_ge", 0.75))
            if float(distress) < float(dist_ge) and float(repeat_ratio) < float(rep_ge):
                st.allow_last = False
                st.deny_last_reason = "not_distressed"
                return False, "not_distressed"

        roi_floor = float(cfg.get("roi_floor", -1.0))
        cold_start_allow = bool(cfg.get("allow_cold_start", True))
        if float(roi) < float(roi_floor) and not (cold_start_allow and int(st.fired) <= 0):
            st.allow_last = False
            st.deny_last_reason = "roi_low"
            return False, "roi_low"

        st.allow_last = True
        st.deny_last_reason = ""
        return True, ""

    def fired(self, comp: str, step: int) -> None:
        st = self.states.get(str(comp))
        if st is None:
            return
        st.last_fire_step = int(step)
        st.fired += 1

    def observe(self, comp: str, step: int, success: bool, roi: float) -> None:
        comp = str(comp)
        st = self.states.get(comp)
        if st is None:
            return
        cfg = self._get_cfg(comp)
        cd_fail = int(cfg.get("cooldown_fail", 0))
        max_fail = int(cfg.get("max_fail_streak", 9999))
        step = int(step)

        if success:
            st.fail_streak = 0
            return

        st.fail_streak += 1
        if st.fail_streak >= max(1, max_fail):
            st.cooldown_until = max(int(st.cooldown_until), step + max(0, cd_fail))
            st.fail_streak = 0

        if comp == "mem":
            self._mem_hist.append(0)
            if self.mem_window > 0 and len(self._mem_hist) >= self.mem_window:
                win = self._mem_hist[-self.mem_window :]
                fail_rate = 1.0 - float(sum(win)) / float(max(1, len(win)))
                if fail_rate >= float(self.mem_fail_rate_hi) and float(roi) <= float(self.mem_roi_floor):
                    self.mem_global_until = max(int(self.mem_global_until), step + int(self.mem_global_cooldown))

    def observe_mem_success(self, step: int) -> None:
        self._mem_hist.append(1)
        if self.mem_window > 0 and len(self._mem_hist) > 4 * self.mem_window:
            self._mem_hist = self._mem_hist[-4 * self.mem_window :]

    def quota(self, comp: str, base_quota: int, roi: float) -> int:
        comp = str(comp)
        st = self.states.get(comp)
        if st is None:
            return int(base_quota)
        cfg = self._get_cfg(comp)
        qmax = int(cfg.get("quota_max", base_quota))
        qmin = int(cfg.get("quota_min", 0))
        if not st.allow_last:
            return 0
        q = int(base_quota)
        boost_hi = float(cfg.get("quota_boost_roi", 0.0))
        if float(boost_hi) > 0.0 and float(roi) >= float(boost_hi):
            q += 1
        return int(max(qmin, min(qmax, q)))

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mem_global_until": int(self.mem_global_until),
            "mem_hist_len": int(len(self._mem_hist)),
        }
        for k, st in self.states.items():
            out[str(k)] = {
                "enabled": int(bool(st.enabled)),
                "allow_last": int(bool(st.allow_last)),
                "deny_last_reason": str(st.deny_last_reason or ""),
                "last_fire_step": int(st.last_fire_step),
                "cooldown_until": int(st.cooldown_until),
                "fail_streak": int(st.fail_streak),
                "fired": int(st.fired),
            }
        return out

