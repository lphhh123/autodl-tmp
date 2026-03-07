"""MPVS Controller (v2): budget-stage-aware, share-aware, horizon-aware.

This controller is designed to reduce parameter dependence while enabling:
  - Low-budget suppression (avoid fixed-tax negative effects)
  - High-budget release (use components whose measured ROI is positive)

Key ideas:
  1) Budget-stage aware: progress = used_calls / total_budget
  2) Share-aware: compare component call_share vs gain_share under EWMA
  3) Horizon-aware credit: delayed ROI for trajectory-shift operators (macro/memory)

The controller never calls the evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


@dataclass
class _EwmaAgg:
    calls: float = 0.0
    gain: float = 0.0
    n: int = 0
    roi_long: float = 0.0  # EWMA of delayed ROI (gain per call-span)


@dataclass
class _Ticket:
    src: str
    start_calls: int
    start_best_total: float
    expire_calls: int
    ctx_key: str = ""
    family: str = ""
    sponsored: bool = False


@dataclass
class _HeurAgg:
    gain: float = 0.0
    calls: float = 0.0
    rate: float = 0.0


@dataclass
class _CtxAgg:
    probe_n: int = 0
    probe_pass_heur: int = 0
    probe_pass_cur: int = 0
    probe_margin_heur: float = 0.0
    probe_margin_cur: float = 0.0
    probe_calls: float = 0.0
    trial_sponsored: int = 0
    trial_won: int = 0
    roi_long: float = 0.0
    released: int = 0
    release_hits: int = 0
    last_trial_step: int = -10**9
    last_release_step: int = -10**9


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


class MPVSController:
    """Unified multi-component controller."""

    def __init__(self, cfg: Dict[str, Any], instance_tag: str = "") -> None:
        self.cfg = cfg or {}
        self.instance_tag = str(instance_tag or "")

        self.states: Dict[str, CompState] = {
            "macro": CompState("macro"),
            "mem": CompState("mem"),
            "llm": CompState("llm"),
        }
        self.alpha = float(self.cfg.get("ewma_alpha", 0.2))
        self.agg_total = _EwmaAgg()
        self.agg_by_src: Dict[str, _EwmaAgg] = {k: _EwmaAgg() for k in self.states.keys()}
        self.tickets: List[_Ticket] = []
        self.alpha_long = float(self.cfg.get("ewma_alpha_long", max(0.05, 0.5 * self.alpha)))

        st = (self.cfg.get("budget_stage") or {}) if isinstance(self.cfg, dict) else {}
        self.early_frac = float(st.get("early_frac", 0.20))
        self.late_frac = float(st.get("late_frac", 0.70))

        sh = (self.cfg.get("share") or {}) if isinstance(self.cfg, dict) else {}
        self.share_min_samples = int(sh.get("min_samples", 10))
        self.share_slack_scale = float(sh.get("slack_scale", 0.5))

        self.stagn_norm = float(self.cfg.get("stagn_norm", 20.0))

        memg = (self.cfg.get("mem_global") or {}) if isinstance(self.cfg, dict) else {}
        self.mem_window = int(memg.get("window", 40))
        self.mem_fail_rate_hi = float(memg.get("fail_rate_hi", 0.75))
        self.mem_roi_floor = float(memg.get("roi_floor", 0.0))
        self.mem_global_cooldown = int(memg.get("cooldown_steps", 30))
        self.mem_global_until = 0
        self._mem_hist: List[int] = []

        llm_cfg = (self._get_cfg("llm") or {})
        self.llm_shadow_mode = bool(llm_cfg.get("shadow_mode", True))
        self.llm_shadow_seen = 0
        self.llm_shadow_good = 0

        cec = (self.cfg.get("cec") or {}) if isinstance(self.cfg, dict) else {}
        ctx_cfg = (cec.get("context") or {}) if isinstance(cec, dict) else {}
        self.cec_enabled = bool(cec.get("enabled", True))
        self.cec_probe_min_samples = int(cec.get("probe_min_samples", 6))
        self.cec_trial_max_per_step = int(cec.get("trial_max_per_step", 1))
        self.cec_stagn_ratio_edges = [float(x) for x in (ctx_cfg.get("stagn_ratio_edges", [0.75, 1.50]) or [0.75, 1.50])][:2]
        self.cec_repeat_ratio_edges = [float(x) for x in (ctx_cfg.get("repeat_ratio_edges", [0.55, 0.75]) or [0.55, 0.75])][:2]
        self.cec_blocked_ratio_edges = [float(x) for x in (ctx_cfg.get("blocked_ratio_edges", [0.15, 0.35]) or [0.15, 0.35])][:2]
        if len(self.cec_stagn_ratio_edges) < 2:
            self.cec_stagn_ratio_edges = [0.75, 1.5]
        if len(self.cec_repeat_ratio_edges) < 2:
            self.cec_repeat_ratio_edges = [0.55, 0.75]
        if len(self.cec_blocked_ratio_edges) < 2:
            self.cec_blocked_ratio_edges = [0.15, 0.35]
        self.heur_agg = _HeurAgg()
        self.ctx_agg: Dict[Tuple[str, str, str], _CtxAgg] = {}
        self._sponsor_step = -1
        self._sponsor_count_step = 0

    def _get_cfg(self, comp: str) -> Dict[str, Any]:
        c = self.cfg.get(str(comp), {}) if isinstance(self.cfg, dict) else {}
        return c or {}

    def _budget_progress(self, used_calls: Optional[int], budget_total: Optional[int]) -> float:
        if used_calls is None or budget_total is None:
            return 0.5
        bt = int(budget_total)
        if bt <= 0:
            return 0.5
        return _clamp01(float(int(used_calls)) / float(bt))

    def _stage(self, progress: float) -> str:
        if float(progress) < float(self.early_frac):
            return "early"
        if float(progress) < float(self.late_frac):
            return "mid"
        return "late"

    def _bucket2(self, x: float, e0: float, e1: float) -> int:
        if float(x) < float(e0):
            return 0
        if float(x) < float(e1):
            return 1
        return 2

    def build_context(
        self,
        stagn: int,
        repeat_ratio: float,
        blocked_ratio: float,
        used_calls: Optional[int],
        budget_total: Optional[int],
    ) -> Dict[str, Any]:
        prog = self._budget_progress(used_calls, budget_total)
        stage = self._stage(prog)
        sr = float(stagn) / float(max(1.0, self.stagn_norm))
        stg_b = self._bucket2(sr, self.cec_stagn_ratio_edges[0], self.cec_stagn_ratio_edges[1])
        rep_b = self._bucket2(float(repeat_ratio), self.cec_repeat_ratio_edges[0], self.cec_repeat_ratio_edges[1])
        blk_b = self._bucket2(float(blocked_ratio), self.cec_blocked_ratio_edges[0], self.cec_blocked_ratio_edges[1])
        health_b = max(rep_b, blk_b)
        return {
            "progress": float(prog),
            "stage": str(stage),
            "stagn_bucket": int(stg_b),
            "health_bucket": int(health_b),
            "ctx_key": f"{stage}|stg{stg_b}|hlth{health_b}",
        }

    def _share_ratio(self, comp: str) -> Tuple[float, float, float, float, int]:
        comp = str(comp)
        tot_c = float(self.agg_total.calls)
        tot_g = float(self.agg_total.gain)
        a = self.agg_by_src.get(comp, _EwmaAgg())
        c = float(a.calls)
        g = float(a.gain)
        call_share = c / max(1e-9, tot_c)
        gain_share = g / max(1e-9, tot_g) if tot_g > 0 else 0.0
        ratio = (gain_share + 1e-9) / (call_share + 1e-9)
        n = int(self.agg_total.n)
        slack = float(self.share_slack_scale) / math.sqrt(float(max(1, n)))
        return float(call_share), float(gain_share), float(ratio), float(slack), int(n)

    def _ctx_family_key(self, comp: str, ctx_key: str, family: str) -> Tuple[str, str, str]:
        return (str(comp), str(ctx_key or ""), str(family or ""))

    def _get_ctx_agg(self, comp: str, ctx_key: str, family: str) -> _CtxAgg:
        key = self._ctx_family_key(comp, ctx_key, family)
        if key not in self.ctx_agg:
            self.ctx_agg[key] = _CtxAgg()
        return self.ctx_agg[key]

    def observe_heuristic(self, gain: float, calls: int) -> None:
        g = float(max(0.0, gain))
        c = float(max(1, int(calls)))
        a = float(self.alpha)
        self.heur_agg.gain = (1.0 - a) * float(self.heur_agg.gain) + a * g
        self.heur_agg.calls = (1.0 - a) * float(self.heur_agg.calls) + a * c
        self.heur_agg.rate = float(self.heur_agg.gain) / float(max(1e-9, self.heur_agg.calls))

    def observe_probe(
        self,
        comp: str,
        family: str,
        ctx_key: str,
        margin_heur: float,
        margin_cur: float,
        calls: int,
        pass_heur: bool,
        pass_cur: bool,
    ) -> None:
        a = self._get_ctx_agg(comp, ctx_key, family)
        ew = float(self.alpha)
        a.probe_n += 1
        a.probe_pass_heur += int(bool(pass_heur))
        a.probe_pass_cur += int(bool(pass_cur))
        a.probe_margin_heur = (1.0 - ew) * float(a.probe_margin_heur) + ew * float(margin_heur)
        a.probe_margin_cur = (1.0 - ew) * float(a.probe_margin_cur) + ew * float(margin_cur)
        a.probe_calls = (1.0 - ew) * float(a.probe_calls) + ew * float(max(1, int(calls)))

    def _probe_utility(self, comp: str, ctx_key: str, family: str) -> float:
        a = self._get_ctx_agg(comp, ctx_key, family)
        return float(a.probe_margin_heur) - float(self.heur_agg.rate) * float(a.probe_calls)

    def _probe_slack(self, comp: str, ctx_key: str, family: str) -> float:
        a = self._get_ctx_agg(comp, ctx_key, family)
        return float(self.share_slack_scale) / math.sqrt(float(max(1, a.probe_n)))

    def maybe_sponsor_trial(self, comp: str, family: str, ctx: Optional[Dict[str, Any]], step: int) -> Tuple[bool, str, Dict[str, Any]]:
        c = str(comp)
        ctx0 = ctx or {}
        ctx_key = str(ctx0.get("ctx_key", ""))
        stage = str(ctx0.get("stage", ""))
        a = self._get_ctx_agg(c, ctx_key, family)
        utility = float(self._probe_utility(c, ctx_key, family))
        slack = float(self._probe_slack(c, ctx_key, family))
        meta = {"utility": utility, "slack": slack, "probe_n": int(a.probe_n)}

        if not bool(self.cec_enabled):
            return False, "cec_disabled", meta
        if c != "macro":
            return False, "not_macro", meta
        if stage != "late":
            return False, "not_late", meta
        if int(a.probe_n) < int(self.cec_probe_min_samples):
            return False, "probe_samples_low", meta
        if int(a.probe_pass_heur) <= 0:
            return False, "probe_no_pass_heur", meta
        if float(a.probe_margin_heur) <= 0.0:
            return False, "probe_margin_nonpos", meta
        if float(utility) <= float(slack):
            return False, "utility_not_enough", meta

        stp = int(step)
        if int(self._sponsor_step) != stp:
            self._sponsor_step = stp
            self._sponsor_count_step = 0
        if int(self._sponsor_count_step) >= max(0, int(self.cec_trial_max_per_step)):
            return False, "step_trial_quota", meta
        if int(a.last_trial_step) >= stp - 1:
            return False, "recently_sponsored", meta

        self._sponsor_count_step += 1
        a.trial_sponsored += 1
        a.last_trial_step = stp
        return True, "sponsored", meta

    def tick(self, step: int) -> None:
        for st in self.states.values():
            st.allow_last = False
            st.deny_last_reason = ""
        if self.mem_window > 0 and len(self._mem_hist) > 4 * self.mem_window:
            self._mem_hist = self._mem_hist[-4 * self.mem_window :]
        if self.mem_global_until < 0:
            self.mem_global_until = 0
        if int(self._sponsor_step) != int(step):
            self._sponsor_count_step = 0

    def on_progress(self, used_calls: int, budget_total: int, best_total_seen: float) -> None:
        used_calls = int(used_calls)
        cur_best = float(best_total_seen)
        if not self.tickets:
            return
        kept: List[_Ticket] = []
        for tk in self.tickets:
            if used_calls < int(tk.expire_calls):
                kept.append(tk)
                continue
            span = max(1, used_calls - int(tk.start_calls))
            gain_long = float(max(0.0, float(tk.start_best_total) - cur_best))
            if gain_long > 0.0:
                a = self.agg_by_src.get(str(tk.src))
                if a is not None:
                    al = float(self.alpha_long)
                    roi_long = float(gain_long) / float(span)
                    a.roi_long = (1.0 - al) * float(a.roi_long) + al * float(roi_long)
            if str(tk.src) == "macro" and str(tk.ctx_key) and str(tk.family):
                ca = self._get_ctx_agg("macro", str(tk.ctx_key), str(tk.family))
                al = float(self.alpha_long)
                roi_long_ctx = float(gain_long) / float(span)
                ca.roi_long = (1.0 - al) * float(ca.roi_long) + al * float(roi_long_ctx)
                if bool(tk.sponsored):
                    if float(ca.roi_long) > 0.0 and float(self._probe_utility("macro", tk.ctx_key, tk.family)) >= 0.0:
                        ca.released = 1
                        ca.last_release_step = int(used_calls)
        self.tickets = kept

    def allow(
        self,
        comp: str,
        step: int,
        stagn: int,
        distress: float,
        repeat_ratio: float,
        roi: float,
        used_calls: Optional[int] = None,
        budget_total: Optional[int] = None,
        ctx: Optional[Dict[str, Any]] = None,
        family: str = "",
    ) -> Tuple[bool, str]:
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

        prog = self._budget_progress(used_calls, budget_total)
        stage = self._stage(prog)

        if comp == "llm":
            dist_ge = float(cfg.get("enable_when_distress_ge", 0.7))
            rep_ge = float(cfg.get("enable_when_repeat_ratio_ge", 0.75))
            if float(distress) < float(dist_ge) and float(repeat_ratio) < float(rep_ge):
                st.allow_last = False
                st.deny_last_reason = "not_distressed"
                return False, "not_distressed"
            if stage == "early" and float(distress) < max(float(dist_ge), 0.9):
                st.allow_last = False
                st.deny_last_reason = "budget_early"
                return False, "budget_early"

        rel = self.release_active(comp, family=family, ctx=(ctx or {"stage": stage, "ctx_key": ""}))
        _, _, ratio, slack, n = self._share_ratio(comp)
        if n >= int(self.share_min_samples):
            if float(ratio) < (1.0 - float(slack)):
                if not (comp == "macro" and stage == "late" and (float(self.roi_long("macro")) > 0.0 or rel)):
                    st.allow_last = False
                    st.deny_last_reason = "share_low"
                    return False, "share_low"

        roi_floor = float(cfg.get("roi_floor", -1.0))
        cold_start_allow = bool(cfg.get("allow_cold_start", True))
        if float(roi) < float(roi_floor) and not (cold_start_allow and int(st.fired) <= 0):
            if not (comp == "macro" and stage == "late" and (float(self.roi_long("macro")) > 0.0 or rel)):
                st.allow_last = False
                st.deny_last_reason = "roi_low"
                return False, "roi_low"

        if stage == "early" and comp in {"macro", "mem"}:
            if float(distress) < float(cfg.get("early_distress_ge", 0.9)):
                st.allow_last = False
                st.deny_last_reason = "budget_early"
                return False, "budget_early"

        st.allow_last = True
        st.deny_last_reason = ""
        return True, ""

    def fired(self, comp: str, step: int) -> None:
        st = self.states.get(str(comp))
        if st is None:
            return
        st.last_fire_step = int(step)
        st.fired += 1

    def observe_mem_success(self, step: int) -> None:
        self._mem_hist.append(1)
        if self.mem_window > 0 and len(self._mem_hist) > 4 * self.mem_window:
            self._mem_hist = self._mem_hist[-4 * self.mem_window :]

    def observe(
        self,
        comp: str,
        step: int,
        success: bool,
        roi: float,
        gain: float = 0.0,
        calls: int = 0,
        used_calls: Optional[int] = None,
        budget_total: Optional[int] = None,
        best_total_seen: Optional[float] = None,
        ctx_key: str = "",
        family: str = "",
    ) -> None:
        comp = str(comp)
        st = self.states.get(comp)
        if st is None:
            return
        cfg = self._get_cfg(comp)
        cd_fail = int(cfg.get("cooldown_fail", 0))
        max_fail = int(cfg.get("max_fail_streak", 9999))
        step = int(step)

        g = float(max(0.0, float(gain)))
        c = float(max(0, int(calls)))
        a = float(self.alpha)
        self.agg_total.calls = (1.0 - a) * float(self.agg_total.calls) + a * float(c)
        self.agg_total.gain = (1.0 - a) * float(self.agg_total.gain) + a * float(g)
        self.agg_total.n += 1
        ag = self.agg_by_src.get(comp)
        if ag is None:
            ag = _EwmaAgg()
            self.agg_by_src[comp] = ag
        ag.calls = (1.0 - a) * float(ag.calls) + a * float(c)
        ag.gain = (1.0 - a) * float(ag.gain) + a * float(g)
        ag.n += 1

        if success:
            st.fail_streak = 0
        else:
            st.fail_streak += 1
            if st.fail_streak >= max(1, int(max_fail)):
                st.cooldown_until = max(int(st.cooldown_until), step + max(0, int(cd_fail)))
                st.fail_streak = 0

        if comp == "mem":
            self._mem_hist.append(1 if bool(success) else 0)
            if self.mem_window > 0 and len(self._mem_hist) >= self.mem_window:
                win = self._mem_hist[-self.mem_window :]
                fail_rate = 1.0 - float(sum(win)) / float(max(1, len(win)))
                if fail_rate >= float(self.mem_fail_rate_hi) and float(roi) <= float(self.mem_roi_floor):
                    self.mem_global_until = max(int(self.mem_global_until), step + int(self.mem_global_cooldown))

        if used_calls is not None and budget_total is not None and best_total_seen is not None:
            try:
                self.on_progress(int(used_calls), int(budget_total), float(best_total_seen))
            except Exception:
                pass

    def register_win(
        self,
        comp: str,
        used_calls: int,
        budget_total: int,
        best_total_seen: float,
        ctx_key: str = "",
        family: str = "",
        sponsored: bool = False,
    ) -> None:
        comp = str(comp)
        used_calls = int(used_calls)
        budget_total = int(budget_total)
        best_total_seen = float(best_total_seen)
        horizon = max(300, int(0.05 * float(budget_total))) if budget_total > 0 else 300
        self.tickets.append(
            _Ticket(
                src=comp,
                start_calls=used_calls,
                start_best_total=best_total_seen,
                expire_calls=used_calls + horizon,
                ctx_key=str(ctx_key or ""),
                family=str(family or ""),
                sponsored=bool(sponsored),
            )
        )
        if comp == "macro" and str(ctx_key or "") and str(family or "") and bool(sponsored):
            self._get_ctx_agg("macro", str(ctx_key), str(family)).trial_won += 1

    def roi_long(self, comp: str) -> float:
        ag = self.agg_by_src.get(str(comp))
        return float(ag.roi_long) if ag is not None else 0.0

    def release_active(self, comp: str, family: str = "", ctx: Optional[Dict[str, Any]] = None) -> bool:
        if str(comp) != "macro":
            return False
        ctx0 = ctx or {}
        if str(ctx0.get("stage", "")) != "late":
            return False
        a = self._get_ctx_agg("macro", str(ctx0.get("ctx_key", "")), str(family or ""))
        utility = self._probe_utility("macro", str(ctx0.get("ctx_key", "")), str(family or ""))
        if float(a.roi_long) <= 0.0 and float(utility) <= 0.0:
            a.released = 0
        if bool(a.released):
            a.release_hits += 1
        return bool(a.released)

    def quota(
        self,
        comp: str,
        base_quota: int,
        roi: float,
        used_calls: Optional[int] = None,
        budget_total: Optional[int] = None,
        ctx: Optional[Dict[str, Any]] = None,
        family: str = "",
    ) -> int:
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

        prog = self._budget_progress(used_calls, budget_total)
        stage = self._stage(prog)

        if stage == "early" and comp in {"macro", "mem"}:
            q = min(q, 1)
        if comp == "llm" and self.llm_shadow_mode:
            return 0

        _, _, ratio, slack, n = self._share_ratio(comp)
        if n >= int(self.share_min_samples):
            if float(ratio) < (1.0 - float(slack)):
                q = max(qmin, q - 1)
            elif float(ratio) > (1.0 + float(slack)):
                q = min(qmax, q + 1)

        if comp == "macro" and stage == "late" and float(self.roi_long("macro")) > 0.0:
            q = min(qmax, q + 1)
        if comp == "macro" and self.release_active(comp, family=family, ctx=(ctx or {"stage": stage, "ctx_key": ""})):
            q = min(qmax, q + 1)

        boost_hi = float(cfg.get("quota_boost_roi", 0.0))
        if float(boost_hi) > 0.0 and float(roi) >= float(boost_hi):
            q = min(qmax, q + 1)

        return int(max(qmin, min(qmax, q)))

    def adjust_min_gain_current(
        self,
        comp: str,
        default_min_gain: float,
        used_calls: Optional[int] = None,
        budget_total: Optional[int] = None,
        ctx: Optional[Dict[str, Any]] = None,
        family: str = "",
        sponsored: bool = False,
    ) -> float:
        comp = str(comp)
        mg = float(default_min_gain)
        prog = self._budget_progress(used_calls, budget_total)
        stage = self._stage(prog)
        if comp == "macro" and bool(sponsored):
            return 0.0
        if comp == "macro" and self.release_active(comp, family=family, ctx=(ctx or {"stage": stage, "ctx_key": ""})):
            return 0.0
        if comp == "macro" and stage == "late" and float(self.roi_long("macro")) > 0.0:
            return float(min(max(0.0, mg), 0.0))
        return float(max(0.0, mg))

    def llm_shadow_observe(self, est_gain: float) -> None:
        self.llm_shadow_seen += 1
        if float(est_gain) > 0.0:
            self.llm_shadow_good += 1

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mem_global_until": int(self.mem_global_until),
            "mem_hist_len": int(len(self._mem_hist)),
            "ewma_total_calls": float(self.agg_total.calls),
            "ewma_total_gain": float(self.agg_total.gain),
            "ewma_n": int(self.agg_total.n),
            "llm_shadow_mode": int(bool(self.llm_shadow_mode)),
            "llm_shadow_seen": int(self.llm_shadow_seen),
            "llm_shadow_good": int(self.llm_shadow_good),
            "roi_long": {k: float(self.roi_long(k)) for k in self.states.keys()},
            "cec_enabled": int(bool(self.cec_enabled)),
            "heur_rate_ewma": float(self.heur_agg.rate),
        }
        cec_ctx: Dict[str, Dict[str, Any]] = {}
        rel_total = 0
        probe_total = 0
        for (comp, ctx_key, family), a in self.ctx_agg.items():
            probe_total += int(a.probe_n)
            rel_total += int(a.released)
            k = f"{comp}|{ctx_key}|{family}"
            cec_ctx[k] = {
                "probe_n": int(a.probe_n),
                "probe_pass_heur": int(a.probe_pass_heur),
                "probe_pass_cur": int(a.probe_pass_cur),
                "probe_margin_heur": float(a.probe_margin_heur),
                "probe_margin_cur": float(a.probe_margin_cur),
                "probe_calls": float(a.probe_calls),
                "probe_utility": float(self._probe_utility(comp, ctx_key, family)),
                "probe_slack": float(self._probe_slack(comp, ctx_key, family)),
                "trial_sponsored": int(a.trial_sponsored),
                "trial_won": int(a.trial_won),
                "roi_long": float(a.roi_long),
                "released": int(a.released),
                "release_hits": int(a.release_hits),
            }
        out["cec_probe_total"] = int(probe_total)
        out["cec_release_total"] = int(rel_total)
        out["cec_ctx"] = cec_ctx

        for k, st in self.states.items():
            call_share, gain_share, ratio, slack, n = self._share_ratio(k)
            out[str(k)] = {
                "enabled": int(bool(st.enabled)),
                "allow_last": int(bool(st.allow_last)),
                "deny_last_reason": str(st.deny_last_reason or ""),
                "last_fire_step": int(st.last_fire_step),
                "cooldown_until": int(st.cooldown_until),
                "fail_streak": int(st.fail_streak),
                "fired": int(st.fired),
                "call_share": float(call_share),
                "gain_share": float(gain_share),
                "share_ratio": float(ratio),
                "share_slack": float(slack),
                "roi_long": float(self.roi_long(k)),
            }
        return out
