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
from collections import deque
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
    # BC^2-CEC: counterfactual baseline snapshot at ticket creation
    start_atomic_rate: float = 0.0
    start_real_rate: float = 0.0
    start_cf_discount: float = 1.0
    start_cf_cap_mult: float = 2.0
    start_cf_mode: str = ""
    expected_atomic_gain: float = 0.0
    # v2.8: track best (minimum) total seen during the ticket horizon (extreme-value credit)
    min_best_total: float = 1.0e30
    min_best_calls: int = -1


class _CallWindowAgg:
    """Call-indexed sliding window aggregator.

    Store samples as (end_calls, gain, calls) and keep only those within a call window.
    This is the standard fix for non-stationary credit assignment in operator selection / AOS.
    Returned rate = sum(gain) / sum(calls).
    """

    def __init__(self, window_calls: int) -> None:
        self.window_calls = int(max(50, window_calls))
        self.items = deque()  # (end_calls, gain, calls)
        self.sum_gain = 0.0
        self.sum_calls = 0.0
        self.n = 0

    def _trim(self, now_calls: int) -> None:
        lo = int(now_calls) - int(self.window_calls)
        while self.items and int(self.items[0][0]) < lo:
            _end, _g, _c = self.items.popleft()
            self.sum_gain -= float(_g)
            self.sum_calls -= float(_c)

    def add(self, now_calls: int, gain: float, calls: int) -> None:
        now_calls = int(now_calls)
        g = float(gain)
        c = float(max(1, int(calls)))
        self.items.append((now_calls, g, c))
        self.sum_gain += g
        self.sum_calls += c
        self.n += 1
        self._trim(now_calls)

    @property
    def rate(self) -> float:
        denom = float(max(1e-9, self.sum_calls))
        return float(self.sum_gain) / denom


@dataclass
class _BurstState:
    family: str = ""
    until_calls: int = 0
    chosen_calls: int = 0
    stage: str = ""
    reason: str = ""


@dataclass
class _Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if self.n <= 1:
            self.mean = float(x)
            self.m2 = 0.0
            return
        d = float(x) - float(self.mean)
        self.mean += d / float(self.n)
        d2 = float(x) - float(self.mean)
        self.m2 += d * d2

    @property
    def var(self) -> float:
        if self.n <= 1:
            return 0.0
        return float(self.m2) / float(max(1, self.n - 1))

    def lcb(self, z: float) -> float:
        if self.n <= 1:
            return float(self.mean)
        v = float(max(0.0, self.var))
        se = math.sqrt(v / float(max(1, self.n)))
        return float(self.mean) - float(z) * float(se)


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
    last_trial_kind: str = ""
    # v2.1 audit: why did we promote to released?
    last_release_reason: str = ""
    # BC^2-CEC audit: realized gains at last horizon maturity (for paper tables)
    last_gain_long: float = 0.0
    last_gain_cf: float = 0.0
    last_atomic_exp_gain: float = 0.0
    # v2.1: candidate soft-release state.
    # Activated immediately after a sponsored macro win, to enable a tiny amount of continued
    # sponsored trials before horizon credit is realized (avoids "first win but no release" deadlock).
    candidate: int = 0
    candidate_hits: int = 0
    last_candidate_step: int = -10**9
    # v2.4: keep candidate alive across weak maturities before revoking.
    candidate_mature_failures: int = 0


@dataclass
class _AosTicket:
    """Short-horizon credit ticket for AOS-style family selection.

    AOS needs low-latency credit updates; we maintain a shorter horizon and update
    per-family sliding-window credit when the ticket matures.
    """

    start_calls: int
    start_best_total: float
    expire_calls: int
    stage: str
    ctx_key: str
    family: str
    min_best_total: float = 1.0e30
    min_best_calls: int = -1


@dataclass
class _FamAgg:
    probe_n: int = 0
    probe_pass_heur: int = 0
    probe_pass_cur: int = 0
    probe_margin_heur: float = 0.0
    probe_margin_cur: float = 0.0
    probe_calls: float = 0.0
    trial_seed_used: int = 0
    trial_sponsored: int = 0
    trial_won: int = 0
    roi_long: float = 0.0
    last_trial_step: int = -10**9


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
        self.alpha_long = float(self.cfg.get("ewma_alpha_long", max(0.05, 0.5 * self.alpha)))
        self.agg_total = _EwmaAgg()
        self.agg_by_src: Dict[str, _EwmaAgg] = {k: _EwmaAgg() for k in self.states.keys()}
        self.tickets: List[_Ticket] = []

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
        # credit_metric:
        #   - "gain_long": legacy long credit
        #   - "gain_cf"  : counterfactual long credit (gain_long - expected_atomic_gain)
        self.cec_credit_metric = str(cec.get("credit_metric", "gain_long") or "gain_long").lower()
        self.cec_counterfactual_credit = bool(cec.get("counterfactual_credit", False)) or (self.cec_credit_metric in {"gain_cf", "cf", "counterfactual"})
        self.cec_cf_use_ctx_atomic = bool(cec.get("counterfactual_use_ctx_atomic", True))
        self.cec_cf_alpha = float(cec.get("counterfactual_alpha", self.alpha))
        # BC^2-CEC: discount for expected atomic gain to avoid over-optimistic counterfactual baseline.
        # 1.0 means "no discount" (legacy behavior). Typical useful range: 0.3~0.7.
        self.cec_cf_discount = float(cec.get("counterfactual_discount", 1.0))
        if not (self.cec_cf_discount == self.cec_cf_discount):  # NaN guard
            self.cec_cf_discount = 1.0
        self.cec_cf_discount = float(max(0.0, min(1.0, self.cec_cf_discount)))
        # Cap counterfactual rate by (cap_mult * realized_rate) to avoid over-optimistic atomic baselines.
        self.cec_cf_cap_mult = float(cec.get("counterfactual_cap_mult", 2.0))
        if not (self.cec_cf_cap_mult == self.cec_cf_cap_mult):
            self.cec_cf_cap_mult = 2.0
        self.cec_cf_cap_mult = float(max(0.0, min(10.0, self.cec_cf_cap_mult)))

        # ----------------------------
        # Counterfactual (CEC) extra knobs
        # ----------------------------
        # Stage-wise discount/cap (helps early noise; aligns with your "low suppress high boost" story)
        self.cec_cf_discount_by_stage: Dict[str, float] = {}
        try:
            m = cec.get("counterfactual_discount_by_stage", {})
            if isinstance(m, dict):
                for k, v in m.items():
                    kk = str(k or "").strip().lower()
                    if kk in {"early", "mid", "late"}:
                        fv = float(v)
                        if fv == fv:
                            self.cec_cf_discount_by_stage[kk] = float(max(0.0, min(1.0, fv)))
        except Exception:
            self.cec_cf_discount_by_stage = {}

        self.cec_cf_cap_mult_by_stage: Dict[str, float] = {}
        try:
            m = cec.get("counterfactual_cap_mult_by_stage", {})
            if isinstance(m, dict):
                for k, v in m.items():
                    kk = str(k or "").strip().lower()
                    if kk in {"early", "mid", "late"}:
                        fv = float(v)
                        if fv == fv:
                            self.cec_cf_cap_mult_by_stage[kk] = float(max(0.0, min(10.0, fv)))
        except Exception:
            self.cec_cf_cap_mult_by_stage = {}

        # Gate using realized-rate baseline: require enough accumulated calls (EWMA calls).
        self.cec_cf_min_real_calls = int(cec.get("counterfactual_min_real_calls", 0))
        self.cec_cf_min_real_calls = int(max(0, self.cec_cf_min_real_calls))

        # Baseline mode:
        #  - "atomic":              use atomic (heuristic) rate only
        #  - "atomic_cap_ctx_real": cap atomic by ctx-realized when available (recommended)
        #  - "atomic_cap_real":     cap atomic by ctx-realized else global-realized/fallback
        self.cec_cf_mode = str(cec.get("counterfactual_baseline_mode", "atomic_cap_ctx_real") or "atomic_cap_ctx_real").strip().lower()
        if self.cec_cf_mode not in {"atomic", "atomic_cap_ctx_real", "atomic_cap_real"}:
            self.cec_cf_mode = "atomic_cap_ctx_real"

        # Optional fallback real-rate when ctx-realized is missing (only used in atomic_cap_real).
        self.cec_cf_fallback_real_rate = float(cec.get("counterfactual_fallback_real_rate", 0.0))
        if not (self.cec_cf_fallback_real_rate == self.cec_cf_fallback_real_rate):
            self.cec_cf_fallback_real_rate = 0.0
        self.cec_cf_fallback_real_rate = float(max(0.0, self.cec_cf_fallback_real_rate))
        self.cec_family_blend_tau = float(cec.get("family_blend_tau", 8))
        self.cec_family_min_samples = int(cec.get("family_min_samples", cec.get("probe_min_samples", 6)))
        self.cec_local_min_samples = int(cec.get("local_min_samples", 2))
        self.cec_seed_trials_per_family = int(cec.get("seed_trials_per_family", 1))
        # v2.3: stage-aware sponsor on-ramp
        # - sponsor is allowed in mid+late (instead of late-only)
        # - direct-pass seed is allowed only in late by default
        sponsor_stages = cec.get("sponsor_stages", ["mid", "late"])
        if isinstance(sponsor_stages, str):
            sponsor_stages = [sponsor_stages]
        self.cec_sponsor_stages = tuple(str(x or "").strip().lower() for x in (sponsor_stages or ["mid", "late"]) if str(x or "").strip())

        direct_pass_stages = cec.get("direct_pass_trial_stages", ["late"])
        if isinstance(direct_pass_stages, str):
            direct_pass_stages = [direct_pass_stages]
        self.cec_direct_pass_trial_stages = tuple(str(x or "").strip().lower() for x in (direct_pass_stages or ["late"]) if str(x or "").strip())

        self.cec_seed_trials_per_family_mid = int(cec.get("seed_trials_per_family_mid", max(1, int(self.cec_seed_trials_per_family))))
        self.cec_seed_trials_per_family_late = int(cec.get("seed_trials_per_family_late", max(1, int(self.cec_seed_trials_per_family) + 1)))
        self.cec_direct_pass_seed_trials_per_family = int(cec.get("direct_pass_seed_trials_per_family", 1))
        # v2.2: allow a tiny number of low-sample seed sponsors once a family has shown pass_heur.
        # This is the missing on-ramp for BC^2-CEC: without it, many families never get their first
        # sponsored win, so no ticket / no long credit / no release can ever happen.
        self.cec_seed_allow_low_sample = bool(cec.get("seed_allow_low_sample", True))
        self.cec_seed_min_pass_heur = int(cec.get("seed_min_pass_heur", 1))
        self.cec_family_cooldown_steps = int(cec.get("family_cooldown_steps", 25))
        self.cec_trial_max_per_step = int(cec.get("trial_max_per_step", 1))
        # v2.4: remaining-budget-aware ticket horizon (for sponsored macro only)
        self.cec_ticket_horizon_min = int(cec.get("ticket_horizon_min", 300))
        self.cec_ticket_horizon_frac = float(cec.get("ticket_horizon_frac", 0.02))
        self.cec_ticket_horizon_mid_remaining_frac = float(cec.get("ticket_horizon_mid_remaining_frac", 0.45))
        self.cec_ticket_horizon_late_remaining_frac = float(cec.get("ticket_horizon_late_remaining_frac", 0.25))
        self.cec_ticket_horizon_mid_cap = int(cec.get("ticket_horizon_mid_cap", 6000))
        self.cec_ticket_horizon_late_cap = int(cec.get("ticket_horizon_late_cap", 2500))
        # v2.1: candidate soft-release + stage-aware family priors
        self.cec_candidate_release = bool(cec.get("candidate_release", True))
        self.cec_family_stage_prior = bool(cec.get("family_stage_prior", True))
        # v2.4: soft/sticky release thresholds
        self.cec_release_credit_floor = float(cec.get("release_credit_floor", -1e-4))
        self.cec_release_roi_floor_local = float(cec.get("release_roi_floor_local", 0.0))
        self.cec_release_roi_floor_family = float(cec.get("release_roi_floor_family", 0.0))
        self.cec_release_trial_score_floor = float(cec.get("release_trial_score_floor", -0.02))
        self.cec_release_edge_floor = float(cec.get("release_edge_floor", -0.02))
        self.cec_release_keep_roi_floor_local = float(cec.get("release_keep_roi_floor_local", -1e-6))
        self.cec_release_keep_roi_floor_family = float(cec.get("release_keep_roi_floor_family", -1e-6))
        self.cec_release_keep_trial_score_floor = float(cec.get("release_keep_trial_score_floor", -0.05))
        # v2.4: candidate grace window
        self.cec_candidate_grace_maturities = int(cec.get("candidate_grace_maturities", 2))
        self.cec_candidate_grace_calls_mid = int(cec.get("candidate_grace_calls_mid", 12000))
        self.cec_candidate_grace_calls_late = int(cec.get("candidate_grace_calls_late", 6000))
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
        self.atomic_by_ctx: Dict[str, _HeurAgg] = {}

        # ------------------------------------------------------------------
        # AOS-style family selection (non-stationary, budget-stage conditioned)
        # ------------------------------------------------------------------
        cec = (self.cfg.get("cec") or {}) if isinstance(self.cfg, dict) else {}
        aos = (cec.get("aos") or {}) if isinstance(cec, dict) else {}
        self.aos_enabled = bool(aos.get("enabled", False))
        # Sliding window size in eval-calls.
        self.aos_window_calls = int(aos.get("window_calls", 6000))
        self.aos_min_samples = int(aos.get("min_samples", 2))
        # UCB-style exploration coefficient.
        self.aos_ucb_c = float(aos.get("ucb_c", 0.6))
        # Extreme-value gain inside horizon ("min") or end-point ("end")
        self.aos_gain_mode = str(aos.get("gain_mode", "min") or "min").strip().lower()
        if self.aos_gain_mode not in {"min", "end"}:
            self.aos_gain_mode = "min"
        # Short horizon for AOS maturation (calls)
        self.aos_horizon_min = int(aos.get("horizon_min", 200))
        self.aos_horizon_frac = float(aos.get("horizon_frac", 0.006))
        self.aos_horizon_mid_cap = int(aos.get("horizon_mid_cap", 3200))
        self.aos_horizon_late_cap = int(aos.get("horizon_late_cap", 1600))
        # Conservative counterfactual baseline for AOS credit (avoid suppressing macros)
        self.aos_cf_discount = float(aos.get("cf_discount", 0.35))
        self.aos_cf_cap_mult = float(aos.get("cf_cap_mult", 1.5))
        if not (self.aos_cf_discount == self.aos_cf_discount):
            self.aos_cf_discount = 0.35
        if not (self.aos_cf_cap_mult == self.aos_cf_cap_mult):
            self.aos_cf_cap_mult = 1.5
        self.aos_cf_discount = float(max(0.0, min(1.0, self.aos_cf_discount)))
        self.aos_cf_cap_mult = float(max(0.0, min(10.0, self.aos_cf_cap_mult)))

        # Internal windows:
        # - family credit window: (stage, family) -> credit/call
        # - realized slope window: (stage, ctx_key) -> gain/call of best_total_seen
        self._aos_fam_win: Dict[Tuple[str, str], _CallWindowAgg] = {}
        self._aos_stage_total_n: Dict[str, int] = {"early": 0, "mid": 0, "late": 0}
        self._real_win: Dict[Tuple[str, str], _CallWindowAgg] = {}
        self._real_last: Dict[Tuple[str, str], Tuple[int, float]] = {}
        self._aos_tickets: List[_AosTicket] = []

        self._burst = _BurstState()

        safe = (cec.get("safety") or {}) if isinstance(cec, dict) else {}
        self.safety_mode = str(safe.get("mode", "none") or "none").strip().lower()
        if self.safety_mode not in {"none", "conservative", "conservative_dgate"}:
            self.safety_mode = "none"
        try:
            self.safety_lcb_z = float(safe.get("lcb_z", 1.0))
        except Exception:
            self.safety_lcb_z = 1.0
        self.safety_lcb_z = float(max(0.0, min(3.0, self.safety_lcb_z)))
        self.safety_min_samples = int(max(1, min(50, int(safe.get("min_samples", 3)))))
        try:
            self.safety_lcb_floor = float(safe.get("lcb_floor", 0.0))
        except Exception:
            self.safety_lcb_floor = 0.0
        self._probe_mom: Dict[Tuple[str, str], _Moments] = {}

        pf = (aos.get("probe_feed") or {}) if isinstance(aos, dict) else {}
        self.aos_probe_feed_enabled = bool(pf.get("enabled", False))
        try:
            self.aos_probe_gain_scale = float(pf.get("gain_scale", 0.25))
        except Exception:
            self.aos_probe_gain_scale = 0.25
        self.aos_probe_gain_scale = float(max(0.0, min(2.0, self.aos_probe_gain_scale)))
        self.aos_probe_use_margin = str(pf.get("use_margin", "heur") or "heur").strip().lower()
        if self.aos_probe_use_margin not in {"heur", "cur"}:
            self.aos_probe_use_margin = "heur"
        self.aos_probe_require_pass_heur = bool(pf.get("require_pass_heur", True))
        self.aos_probe_calls_cap = int(max(50, min(20000, int(pf.get("calls_cap", 1200)))))

        # Debounce for on_progress to avoid double-processing at the same used_calls
        self._last_progress_calls = -1
        # v2.6: realized (diminishing-returns-aware) progress rate of best_total_seen, per ctx_key.
        self.real_agg = _HeurAgg()
        self.real_by_ctx: Dict[str, _HeurAgg] = {}
        self._real_last_global: Optional[Tuple[int, float]] = None
        self._real_last_by_ctx: Dict[str, Tuple[int, float]] = {}
        self.ctx_agg: Dict[Tuple[str, str, str], _CtxAgg] = {}
        self.family_agg: Dict[Tuple[str, str], _FamAgg] = {}
        self.family_stage_agg: Dict[Tuple[str, str, str], _FamAgg] = {}
        self._sponsor_step = -1
        self._sponsor_count_step = 0

    def _candidate_grace_calls(self, stage: str) -> int:
        st = str(stage or "").strip().lower()
        if st == "late":
            return max(int(self.cec_ticket_horizon_min), int(self.cec_candidate_grace_calls_late))
        return max(int(self.cec_ticket_horizon_min), int(self.cec_candidate_grace_calls_mid))

    def _soft_release_eval(
        self,
        ctx_key: str,
        family: str,
        credit_gain: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, float]]:
        ca = self._get_ctx_agg("macro", str(ctx_key or ""), str(family or ""))
        fa = self._get_family_agg("macro", str(family or ""))
        ts = self._trial_score("macro", str(ctx_key or ""), str(family or ""))
        roi_local = float(getattr(ca, "roi_long", 0.0))
        roi_family = float(getattr(fa, "roi_long", 0.0))
        trial_score = float(ts.get("trial_score", 0.0))
        edge_local = float(ts.get("edge_local", 0.0))
        edge_family = float(ts.get("edge_family", 0.0))
        trial_won = int(getattr(ca, "trial_won", 0))
        pass_heur = max(int(getattr(ca, "probe_pass_heur", 0)), int(getattr(fa, "probe_pass_heur", 0)))
        meta = {
            "roi_local": roi_local,
            "roi_family": roi_family,
            "trial_score": trial_score,
            "edge_local": edge_local,
            "edge_family": edge_family,
            "credit_gain": float(credit_gain) if credit_gain is not None else 0.0,
        }
        # Rule A (precision-first): realized-credit release.
        # Once a sponsored macro has matured and produced non-negative counterfactual gain,
        # do not let ex-ante scores block release.
        if (
            credit_gain is not None
            and float(credit_gain) >= float(self.cec_release_credit_floor)
            and trial_won > 0
            and pass_heur > 0
        ):
            return True, "gain_cf_realized", meta
        # Rule B: conservative fallback bridge.
        if (
            bool(getattr(ca, "candidate", 0))
            and trial_won > 0
            and (
                roi_local > float(self.cec_release_roi_floor_local)
                or roi_family > float(self.cec_release_roi_floor_family)
            )
        ):
            return True, "candidate_bridge", meta
        return False, "", meta

    def _seed_budget_for_stage(self, stage: str) -> int:
        st = str(stage or "").strip().lower()
        if st == "mid":
            return max(0, int(self.cec_seed_trials_per_family_mid))
        if st == "late":
            return max(0, int(self.cec_seed_trials_per_family_late))
        return max(0, int(self.cec_seed_trials_per_family))

    def _family_stage_key(self, comp: str, stage: str, family: str) -> Tuple[str, str, str]:
        return (str(comp), str(stage or ""), str(family or ""))

    def _get_family_stage_agg(self, comp: str, stage: str, family: str) -> _FamAgg:
        key = self._family_stage_key(comp, stage, family)
        if key not in self.family_stage_agg:
            self.family_stage_agg[key] = _FamAgg()
        return self.family_stage_agg[key]

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
            # v2.5 precision-first:
            # Probe/sponsor may use fine ctx_key, but candidate/release should use a coarser key
            # to avoid over-fragmentation and make successful macro evidence reusable.
            "release_ctx_key": f"{stage}|stg{stg_b}",
        }

    def _active_ctx_key(self, ctx: Optional[Dict[str, Any]]) -> str:
        ctx0 = ctx or {}
        stage = str(ctx0.get("stage", "") or "")
        if stage == "late":
            return str(ctx0.get("release_ctx_key", ctx0.get("ctx_key", "")) or "")
        return str(ctx0.get("ctx_key", "") or "")

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

    def _family_key(self, comp: str, family: str) -> Tuple[str, str]:
        return (str(comp), str(family or ""))

    def _get_family_agg(self, comp: str, family: str) -> _FamAgg:
        key = self._family_key(comp, family)
        if key not in self.family_agg:
            self.family_agg[key] = _FamAgg()
        return self.family_agg[key]

    def observe_heuristic(self, gain: float, calls: int) -> None:
        g = float(max(0.0, gain))
        c = float(max(1, int(calls)))
        a = float(self.alpha)
        self.heur_agg.gain = (1.0 - a) * float(self.heur_agg.gain) + a * g
        self.heur_agg.calls = (1.0 - a) * float(self.heur_agg.calls) + a * c
        self.heur_agg.rate = float(self.heur_agg.gain) / float(max(1e-9, self.heur_agg.calls))

    def observe_atomic(self, ctx_key: str, gain: float, calls: int) -> None:
        """BC^2-CEC: track atomic counterfactual baseline rate per ctx_key.

        We use the *best heuristic (atomic)* candidate at this step as the counterfactual
        "what we could have achieved using only atomic moves".
        """
        ctx_key = str(ctx_key or "")
        if not ctx_key:
            return
        g = float(max(0.0, gain))
        c = float(max(1, int(calls)))
        ew = float(self.cec_cf_alpha)
        ag = self.atomic_by_ctx.get(ctx_key)
        if ag is None:
            ag = _HeurAgg()
            self.atomic_by_ctx[ctx_key] = ag
        ag.gain = (1.0 - ew) * float(ag.gain) + ew * g
        ag.calls = (1.0 - ew) * float(ag.calls) + ew * c
        ag.rate = float(ag.gain) / float(max(1e-9, ag.calls))

    def _atomic_rate(self, ctx_key: str = "") -> float:
        """Return per-ctx atomic baseline if available; else fall back to global heuristic rate."""
        ctx_key = str(ctx_key or "")
        if bool(self.cec_cf_use_ctx_atomic) and ctx_key:
            ag = self.atomic_by_ctx.get(ctx_key)
            if ag is not None and float(ag.calls) > 0.0:
                return float(ag.rate)
        return float(self.heur_agg.rate)

    # ----------------------------
    # AOS: non-stationary baselines
    # ----------------------------
    def _get_win(self, store: Dict[Tuple[str, str], _CallWindowAgg], key: Tuple[str, str], window_calls: int) -> _CallWindowAgg:
        w = store.get(key)
        if w is None:
            w = _CallWindowAgg(int(window_calls))
            store[key] = w
        return w

    def _stage_from_ctx_key(self, ctx_key: str) -> str:
        s = str(ctx_key or "")
        if not s:
            return ""
        st = s.split("|", 1)[0].strip().lower()
        return st if st in {"early", "mid", "late"} else ""

    def observe_best_total(self, ctx_key: str, stage: str, used_calls: int, best_total_seen: float) -> None:
        """Update realized atomic baseline via sliding-window slope of best_total_seen."""
        stage = str(stage or "").strip().lower()
        if stage not in {"early", "mid", "late"}:
            return
        ck = str(ctx_key or "").strip()
        if not ck:
            return
        u = int(used_calls)
        b = float(best_total_seen)
        last = self._real_last.get((stage, ck))
        if last is None:
            self._real_last[(stage, ck)] = (u, b)
            return
        lu, lb = int(last[0]), float(last[1])
        dc = int(u) - int(lu)
        if dc <= 0:
            return
        dg = float(lb) - float(b)
        self._real_last[(stage, ck)] = (u, b)
        if dg <= 0.0:
            return
        w = self._get_win(self._real_win, (stage, ck), int(self.aos_window_calls))
        w.add(now_calls=u, gain=float(dg), calls=int(dc))

    def _aos_realized_rate(self, ctx_key: str, stage: str) -> float:
        stage = str(stage or "").strip().lower()
        ck = str(ctx_key or "").strip()
        if stage in {"early", "mid", "late"} and ck:
            w = self._real_win.get((stage, ck))
            if w is not None and float(w.sum_calls) > 0.0:
                return float(w.rate)
        return 0.0

    def _aos_score(self, family: str, stage: str) -> Tuple[float, int]:
        stage = str(stage or "").strip().lower()
        fam = str(family or "").strip()
        w = self._aos_fam_win.get((stage, fam))
        if w is None:
            return 0.0, 0
        return float(w.rate), int(w.n)

    def rank_macro_families(self, families: List[str], ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        """Rank macro families for proposing, using AOS-style sliding-window credit + UCB bonus."""
        if not bool(self.aos_enabled):
            return list(families or [])
        fams = [str(x or "").strip() for x in (families or []) if str(x or "").strip()]
        if len(fams) <= 1:
            return fams
        ctx0 = ctx or {}
        stage = str(ctx0.get("stage", "") or "").strip().lower()
        if stage not in {"early", "mid", "late"}:
            stage = self._stage_from_ctx_key(str(ctx0.get("ctx_key", "") or ""))
        if stage not in {"early", "mid", "late"}:
            return fams

        try:
            used_calls = int(ctx0.get("used_calls", -1))
        except Exception:
            used_calls = -1
        if used_calls >= 0:
            bf = self.get_burst_family(int(used_calls))
            if bf and bf in fams:
                return [bf]

        fams = self.filter_families_conservative(fams, stage)
        if len(fams) <= 1:
            return fams

        t = int(self._aos_stage_total_n.get(stage, 0))
        scored: List[Tuple[float, int, str]] = []
        for i, f in enumerate(fams):
            s, n = self._aos_score(f, stage)
            bonus = 0.0
            if float(self.aos_ucb_c) > 0.0:
                bonus = float(self.aos_ucb_c) * math.sqrt(math.log(float(1 + t)) / float(1 + n))
            scored.append((float(s) + float(bonus), int(i), f))
        # stable: tie keeps original order
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [f for _, __, f in scored]

    def observe_realized(self, ctx_key: str, used_calls: int, best_total_seen: float) -> None:
        """Track realized improvement rate of best_total_seen per eval-call for this ctx_key."""
        ck = str(ctx_key or "")
        if not ck:
            return
        u = int(used_calls)
        b = float(best_total_seen)
        if u < 0 or not (b == b):
            return

        def _upd(agg: _HeurAgg, last: Tuple[int, float]) -> Tuple[int, float]:
            lu, lb = int(last[0]), float(last[1])
            dc = int(u) - int(lu)
            if dc <= 0:
                return (lu, lb)
            dg = float(lb) - float(b)
            if dg <= 0.0:
                return (int(u), float(b))
            al = float(self.alpha_long)
            agg.gain = (1.0 - al) * float(agg.gain) + al * float(dg)
            agg.calls = (1.0 - al) * float(agg.calls) + al * float(dc)
            agg.rate = float(agg.gain) / float(max(1e-9, agg.calls))
            return (int(u), float(b))

        if self._real_last_global is None:
            self._real_last_global = (int(u), float(b))
        else:
            self._real_last_global = _upd(self.real_agg, self._real_last_global)

        last_ck = self._real_last_by_ctx.get(ck)
        if last_ck is None:
            self._real_last_by_ctx[ck] = (int(u), float(b))
            return
        ag = self.real_by_ctx.get(ck)
        if ag is None:
            ag = _HeurAgg()
            self.real_by_ctx[ck] = ag
        self._real_last_by_ctx[ck] = _upd(ag, last_ck)

    def _realized_rate(self, ctx_key: str = "") -> float:
        ck = str(ctx_key or "")
        if ck:
            ag = self.real_by_ctx.get(ck)
            if ag is not None and float(ag.calls) > 0.0:
                return float(ag.rate)
        return float(self.real_agg.rate)

    def _cf_stage_from_ctx_key(self, ctx_key: str) -> str:
        s = str(ctx_key or "")
        if not s:
            return ""
        st = s.split("|", 1)[0].strip().lower()
        return st if st in {"early", "mid", "late"} else ""

    def _cf_discount_for_stage(self, stage: str) -> float:
        st = str(stage or "").strip().lower()
        if st in self.cec_cf_discount_by_stage:
            return float(self.cec_cf_discount_by_stage[st])
        return float(self.cec_cf_discount)

    def _cf_cap_for_stage(self, stage: str) -> float:
        st = str(stage or "").strip().lower()
        if st in self.cec_cf_cap_mult_by_stage:
            return float(self.cec_cf_cap_mult_by_stage[st])
        return float(self.cec_cf_cap_mult)

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
        af = self._get_family_agg(comp, family)
        stage = str(ctx_key or "").split("|", 1)[0] if str(ctx_key or "") else ""
        ew = float(self.alpha)
        for ag in (a, af):
            ag.probe_n += 1
            ag.probe_pass_heur += int(bool(pass_heur))
            ag.probe_pass_cur += int(bool(pass_cur))
            ag.probe_margin_heur = (1.0 - ew) * float(ag.probe_margin_heur) + ew * float(margin_heur)
            ag.probe_margin_cur = (1.0 - ew) * float(ag.probe_margin_cur) + ew * float(margin_cur)
            ag.probe_calls = (1.0 - ew) * float(ag.probe_calls) + ew * float(max(1, int(calls)))

        # v2.1: stage-aware family priors (avoid early negatives diluting late priors)
        if bool(self.cec_family_stage_prior) and stage in {"early", "mid", "late"}:
            asg = self._get_family_stage_agg(comp, stage, family)
            asg.probe_n += 1
            asg.probe_pass_heur += int(bool(pass_heur))
            asg.probe_pass_cur += int(bool(pass_cur))
            asg.probe_margin_heur = (1.0 - ew) * float(asg.probe_margin_heur) + ew * float(margin_heur)
            asg.probe_margin_cur = (1.0 - ew) * float(asg.probe_margin_cur) + ew * float(margin_cur)
            asg.probe_calls = (1.0 - ew) * float(asg.probe_calls) + ew * float(max(1, int(calls)))

        try:
            denom = float(max(1, int(calls)))
            u = float(margin_heur) / denom
            key = (str(stage or ""), str(family or ""))
            if key[0] in {"early", "mid", "late"} and key[1]:
                mm = self._probe_mom.get(key)
                if mm is None:
                    mm = _Moments()
                    self._probe_mom[key] = mm
                mm.add(float(u))
        except Exception:
            pass

    def burst_active(self, used_calls: int) -> bool:
        try:
            return bool(self._burst.family) and int(used_calls) < int(self._burst.until_calls)
        except Exception:
            return False

    def get_burst_family(self, used_calls: int) -> str:
        if self.burst_active(int(used_calls)):
            return str(self._burst.family or "")
        return ""

    def set_burst_family(self, family: str, used_calls: int, burst_calls: int, *, stage: str = "", reason: str = "") -> None:
        fam = str(family or "").strip()
        if not fam:
            return
        used_calls = int(max(0, int(used_calls)))
        burst_calls = int(max(50, int(burst_calls)))
        self._burst.family = fam
        self._burst.chosen_calls = used_calls
        self._burst.until_calls = used_calls + burst_calls
        self._burst.stage = str(stage or "")
        self._burst.reason = str(reason or "")

    def _conservative_keep(self, family: str, stage: str) -> bool:
        if str(self.safety_mode) not in {"conservative", "conservative_dgate"}:
            return True
        st = str(stage or "").strip().lower()
        fam = str(family or "").strip()
        if st not in {"early", "mid", "late"} or not fam:
            return True
        mm = self._probe_mom.get((st, fam))
        if mm is None or int(mm.n) < int(self.safety_min_samples):
            return True
        lcb = float(mm.lcb(float(self.safety_lcb_z)))
        return bool(lcb >= float(self.safety_lcb_floor))

    def filter_families_conservative(self, families: List[str], stage: str) -> List[str]:
        fams = [str(x or "").strip() for x in (families or []) if str(x or "").strip()]
        if not fams:
            return []
        if str(self.safety_mode) not in {"conservative", "conservative_dgate"}:
            return fams
        kept = [f for f in fams if self._conservative_keep(f, stage)]
        return kept if kept else fams

    def observe_probe_credit_to_aos(
        self,
        *,
        family: str,
        stage: str,
        used_calls: int,
        margin_heur: float,
        margin_cur: float,
        calls: int,
        pass_heur: bool = True,
        source: str = "probe",
    ) -> None:
        _ = source
        if not bool(self.aos_enabled) or not bool(self.aos_probe_feed_enabled):
            return
        if self.aos_probe_require_pass_heur and not bool(pass_heur):
            return
        st = str(stage or "").strip().lower()
        fam = str(family or "").strip()
        if st not in {"early", "mid", "late"} or not fam:
            return
        u = int(max(0, int(used_calls)))
        c = int(min(max(1, int(calls)), int(self.aos_probe_calls_cap)))
        g_raw = float(margin_heur) if self.aos_probe_use_margin == "heur" else float(margin_cur)
        g = float(max(0.0, float(g_raw)))
        g = float(self.aos_probe_gain_scale) * float(g)
        w = self._aos_fam_win.get((st, fam))
        if w is None:
            w = _CallWindowAgg(int(self.aos_window_calls))
            self._aos_fam_win[(st, fam)] = w
        w.add(now_calls=u, gain=float(g), calls=int(c))

    def _probe_utility_from(self, margin_heur: float, calls: float) -> float:
        return float(margin_heur) - float(self.heur_agg.rate) * float(calls)

    def _probe_edge_from(self, margin_heur: float, calls: float) -> float:
        utility = self._probe_utility_from(float(margin_heur), float(calls))
        denom = max(1e-12, abs(float(margin_heur)) + float(self.heur_agg.rate) * float(calls) + 1e-12)
        return float(utility) / float(denom)

    def _probe_utility(self, comp: str, ctx_key: str, family: str) -> float:
        a = self._get_ctx_agg(comp, ctx_key, family)
        return self._probe_utility_from(float(a.probe_margin_heur), float(a.probe_calls))

    def _probe_edge_ctx(self, comp: str, ctx_key: str, family: str) -> float:
        a = self._get_ctx_agg(comp, ctx_key, family)
        return self._probe_edge_from(float(a.probe_margin_heur), float(a.probe_calls))

    def _probe_edge_family(self, comp: str, family: str) -> float:
        a = self._get_family_agg(comp, family)
        return self._probe_edge_from(float(a.probe_margin_heur), float(a.probe_calls))

    def _probe_edge_family_stage(self, comp: str, stage: str, family: str) -> float:
        a = self._get_family_stage_agg(comp, stage, family)
        return self._probe_edge_from(float(a.probe_margin_heur), float(a.probe_calls))

    def _edge_family_prior(self, comp: str, stage: str, family: str) -> Dict[str, Any]:
        """Stage-aware family prior for sponsor scoring.

        - In late stage: prefer a mid+late weighted prior (mitigates early negative dilution).
        - If stage samples are insufficient: blend stage prior with global prior using family_blend_tau.
        """
        comp = str(comp)
        stage = str(stage or "")
        family = str(family or "")

        ag_g = self._get_family_agg(comp, family)
        n_g = int(ag_g.probe_n)
        edge_g = float(self._probe_edge_family(comp, family))

        n_m = 0
        n_l = 0
        edge_m = 0.0
        edge_l = 0.0
        if bool(self.cec_family_stage_prior):
            try:
                ag_m = self._get_family_stage_agg(comp, "mid", family)
                n_m = int(ag_m.probe_n)
                edge_m = float(self._probe_edge_family_stage(comp, "mid", family)) if n_m > 0 else 0.0
            except Exception:
                n_m, edge_m = 0, 0.0
            try:
                ag_l = self._get_family_stage_agg(comp, "late", family)
                n_l = int(ag_l.probe_n)
                edge_l = float(self._probe_edge_family_stage(comp, "late", family)) if n_l > 0 else 0.0
            except Exception:
                n_l, edge_l = 0, 0.0

        prior_src = "global_fallback"
        edge_prior = float(edge_g)
        tau = max(1e-9, float(self.cec_family_blend_tau))
        min_samp = int(self.cec_family_min_samples)

        if stage == "late" and bool(self.cec_family_stage_prior):
            n_stage = int(n_m + n_l)
            if n_stage > 0:
                edge_stage = (float(n_m) * float(edge_m) + float(n_l) * float(edge_l)) / float(max(1, n_stage))
                if n_stage >= min_samp:
                    edge_prior = float(edge_stage)
                    prior_src = "midlate"
                else:
                    lam = float(n_stage) / float(float(n_stage) + tau)
                    edge_prior = float(lam) * float(edge_stage) + (1.0 - float(lam)) * float(edge_g)
                    prior_src = "midlate+global"
            else:
                edge_prior = float(edge_g)
                prior_src = "global_fallback"
        elif stage in {"early", "mid"} and bool(self.cec_family_stage_prior):
            ag_s = self._get_family_stage_agg(comp, stage, family)
            n_s = int(ag_s.probe_n)
            if n_s > 0:
                edge_s = float(self._probe_edge_family_stage(comp, stage, family))
                if n_s >= min_samp:
                    edge_prior = float(edge_s)
                    prior_src = "stage"
                else:
                    lam = float(n_s) / float(float(n_s) + tau)
                    edge_prior = float(lam) * float(edge_s) + (1.0 - float(lam)) * float(edge_g)
                    prior_src = "stage+global"
            else:
                edge_prior = float(edge_g)
                prior_src = "global_fallback"

        return {
            "edge_family_prior": float(edge_prior),
            "edge_family_global": float(edge_g),
            "edge_family_mid": float(edge_m),
            "edge_family_late": float(edge_l),
            "n_family_global": int(n_g),
            "n_family_mid": int(n_m),
            "n_family_late": int(n_l),
            "prior_src": str(prior_src),
        }

    def _trial_score(self, comp: str, ctx_key: str, family: str) -> Dict[str, Any]:
        a_local = self._get_ctx_agg(comp, ctx_key, family)
        a_fam = self._get_family_agg(comp, family)
        n_local = int(a_local.probe_n)
        n_family = int(a_fam.probe_n)
        edge_local = float(self._probe_edge_ctx(comp, ctx_key, family))
        stage = str(ctx_key or "").split("|", 1)[0] if str(ctx_key or "") else ""
        prior = self._edge_family_prior(comp, stage, family) if bool(self.cec_family_stage_prior) else {
            "edge_family_prior": float(self._probe_edge_family(comp, family)),
            "edge_family_global": float(self._probe_edge_family(comp, family)),
            "edge_family_mid": 0.0,
            "edge_family_late": 0.0,
            "n_family_global": int(n_family),
            "n_family_mid": 0,
            "n_family_late": 0,
            "prior_src": "global_fallback",
        }
        edge_family = float(prior.get("edge_family_prior", 0.0))
        tau = max(1e-9, float(self.cec_family_blend_tau))
        lambda_local = float(n_local) / float(float(n_local) + tau)
        trial_score = float(lambda_local) * edge_local + (1.0 - float(lambda_local)) * edge_family
        return {
            "edge_local": float(edge_local),
            "edge_family": float(edge_family),
            "edge_family_global": float(prior.get("edge_family_global", 0.0)),
            "edge_family_mid": float(prior.get("edge_family_mid", 0.0)),
            "edge_family_late": float(prior.get("edge_family_late", 0.0)),
            "trial_score": float(trial_score),
            "lambda_local": float(lambda_local),
            "n_local": int(n_local),
            "n_family": int(n_family),
            "n_family_global": int(prior.get("n_family_global", n_family)),
            "n_family_mid": int(prior.get("n_family_mid", 0)),
            "n_family_late": int(prior.get("n_family_late", 0)),
            "prior_src": str(prior.get("prior_src", "")),
        }

    def candidate_active(self, comp: str, family: str = "", ctx: Optional[Dict[str, Any]] = None) -> bool:
        """Soft-release state (candidate).

        Candidate is activated immediately after a sponsored macro win, and expires when the
        corresponding horizon ticket matures (promoted to released if long ROI is positive; otherwise revoked).
        """
        if not bool(self.cec_enabled):
            return False
        if str(comp) != "macro":
            return False
        ctx0 = ctx or {}
        if str(ctx0.get("stage", "")) != "late":
            return False
        if not bool(self.cec_candidate_release):
            return False
        a = self._get_ctx_agg("macro", self._active_ctx_key(ctx0), str(family or ""))
        if bool(a.released):
            return False
        if int(getattr(a, "candidate", 0)) != 1:
            return False
        a.candidate_hits += 1
        return True

    def get_active_families(self, ctx_key: str, stage: str = "", count_hits: bool = True) -> List[str]:
        """Return macro families that are currently active (released or candidate) in this ctx.

        This is intentionally ctx-local (ctx_key + family). It is used to:
          - prioritize macro proposing (try active families first)
          - enable source-level gating when family is not yet chosen

        Notes:
          - Only meaningful in late stage; early/mid return [].
          - When count_hits=True, this method increments per-family hits so trace can show it was used.
        """
        if not bool(self.cec_enabled):
            return []
        ctx_key = str(ctx_key or "")
        stage = str(stage or "")
        if not ctx_key:
            return []
        if stage and stage != "late":
            return []

        items: List[Tuple[int, int, int, int, str]] = []
        for (comp, ck, fam), a in self.ctx_agg.items():
            if str(comp) != "macro":
                continue
            if str(ck) != ctx_key:
                continue
            fam = str(fam or "")
            if not fam:
                continue
            rel = int(getattr(a, "released", 0)) == 1
            cand = bool(self.cec_candidate_release) and int(getattr(a, "candidate", 0)) == 1
            if not (rel or cand):
                continue
            if bool(count_hits):
                if rel:
                    a.release_hits += 1
                if cand:
                    a.candidate_hits += 1
            items.append(
                (
                    1 if rel else 0,
                    1 if cand else 0,
                    int(getattr(a, "last_release_step", -10**9)),
                    int(getattr(a, "last_candidate_step", -10**9)),
                    fam,
                )
            )

        # Priority: released > candidate, then most recent.
        items.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, __, ___, ____, fam in items:
            if fam in seen:
                continue
            seen.add(fam)
            out.append(fam)
        return out

    def candidate_any(self, comp: str, ctx: Optional[Dict[str, Any]] = None) -> bool:
        """Source-level check: any active (released/candidate) macro family exists in this ctx.

        allow()/quota() are called before choosing a concrete macro family, so family-specific
        candidate_active(...) cannot trigger when family=="".
        """
        if str(comp) != "macro":
            return False
        ctx0 = ctx or {}
        if str(ctx0.get("stage", "")) != "late":
            return False
        ctx_key = self._active_ctx_key(ctx0)
        if not ctx_key:
            return False
        fams = self.get_active_families(ctx_key=ctx_key, stage="late", count_hits=True)
        return bool(fams)

    def release_any(self, comp: str, ctx: Optional[Dict[str, Any]] = None) -> bool:
        if str(comp) != "macro":
            return False
        ctx0 = ctx or {}
        if str(ctx0.get("stage", "")) != "late":
            return False
        ctx_key = self._active_ctx_key(ctx0)
        if not ctx_key:
            return False
        for (c, ck, fam), a in self.ctx_agg.items():
            if str(c) != "macro":
                continue
            if str(ck) != ctx_key:
                continue
            if bool(getattr(a, "released", 0)):
                a.release_hits += 1
                return True
        return False

    def maybe_sponsor_trial(self, comp: str, family: str, ctx: Optional[Dict[str, Any]], step: int) -> Tuple[bool, str, Dict[str, Any]]:
        c = str(comp)
        ctx0 = ctx or {}
        ctx_key = self._active_ctx_key(ctx0)
        stage = str(ctx0.get("stage", "")).strip().lower()
        a = self._get_ctx_agg(c, ctx_key, family)
        af = self._get_family_agg(c, family)
        score_meta = self._trial_score(c, ctx_key, family)
        meta = dict(score_meta)

        if not bool(self.cec_enabled):
            return False, "cec_disabled", meta
        if c != "macro":
            return False, "not_macro", meta
        if stage not in set(self.cec_sponsor_stages):
            return False, "not_sponsor_stage", meta
        pass_heur_local = int(a.probe_pass_heur)
        pass_heur_family = int(af.probe_pass_heur)
        pass_heur_need = max(1, int(self.cec_seed_min_pass_heur))
        if pass_heur_local < pass_heur_need and pass_heur_family < pass_heur_need:
            return False, "no_pass_heur", meta

        stp = int(step)
        if int(self._sponsor_step) != stp:
            self._sponsor_step = stp
            self._sponsor_count_step = 0
        if int(self._sponsor_count_step) >= max(0, int(self.cec_trial_max_per_step)):
            return False, "step_trial_quota", meta
        if int(af.last_trial_step) > stp - max(0, int(self.cec_family_cooldown_steps)):
            return False, "family_cooldown", meta

        # v2.1: candidate sponsor path.
        # If a ctx-family has already achieved a sponsored win (candidate==1), allow a tiny amount of
        # continued sponsored trials (still requires pass_heur, still respects per-step quota + family cooldown).
        if bool(self.cec_candidate_release) and self.candidate_active("macro", family=family, ctx=ctx0):
            reason = "candidate_sponsor"
            self._sponsor_count_step += 1
            a.trial_sponsored += 1
            a.last_trial_step = stp
            a.last_trial_kind = str(reason)
            af.trial_sponsored += 1
            af.last_trial_step = stp
            return True, reason, meta

        reason = ""
        trial_score = float(score_meta.get("trial_score", 0.0))
        edge_family = float(score_meta.get("edge_family", 0.0))
        n_local = int(score_meta.get("n_local", 0))
        n_family = int(score_meta.get("n_family", 0))
        seed_budget = int(self._seed_budget_for_stage(stage))

        if n_family >= int(self.cec_family_min_samples) and n_local >= int(self.cec_local_min_samples) and trial_score > 0.0:
            reason = "evidence_sponsor"
        # v2.2: true low-sample seeding path.
        # If a family has already shown pass_heur but still lacks enough samples,
        # allow a very small number of seed sponsors so it can collect its first realized long credit.
        elif (
            bool(self.cec_seed_allow_low_sample)
            and n_family < int(self.cec_family_min_samples)
            and (pass_heur_local >= pass_heur_need or pass_heur_family >= pass_heur_need)
        ):
            if int(af.trial_seed_used) >= max(0, seed_budget):
                return False, "seed_budget_exhausted", meta
            reason = "seed_sponsor_low_sample"
        elif n_family >= int(self.cec_family_min_samples) and edge_family > 0.0 and pass_heur_local > 0:
            if int(af.trial_seed_used) >= max(0, seed_budget):
                return False, "seed_budget_exhausted", meta
            reason = "seed_sponsor"
        else:
            if n_family < int(self.cec_family_min_samples):
                return False, "family_samples_low", meta
            return False, "trial_score_nonpos", meta

        self._sponsor_count_step += 1
        a.trial_sponsored += 1
        a.last_trial_step = stp
        a.last_trial_kind = str(reason)
        af.trial_sponsored += 1
        af.last_trial_step = stp
        if reason in {"seed_sponsor", "seed_sponsor_low_sample"}:
            af.trial_seed_used += 1
        return True, reason, meta

    def maybe_direct_pass_trial(self, comp: str, family: str, ctx: Optional[Dict[str, Any]], step: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Allow a tiny number of direct-pass macro trials into BC^2-CEC.

        This closes the current gap where a macro that already passes current gate
        is selected as a normal macro, but never gets marked as a sponsored trial,
        so std-budgetaware and bc2cec remain identical.
        """
        c = str(comp)
        ctx0 = ctx or {}
        ctx_key = self._active_ctx_key(ctx0)
        stage = str(ctx0.get("stage", "")).strip().lower()
        a = self._get_ctx_agg(c, ctx_key, family)
        af = self._get_family_agg(c, family)
        score_meta = self._trial_score(c, ctx_key, family)
        meta = dict(score_meta)

        if not bool(self.cec_enabled):
            return False, "cec_disabled", meta
        if c != "macro":
            return False, "not_macro", meta
        if stage not in set(self.cec_direct_pass_trial_stages):
            return False, "not_direct_pass_stage", meta

        pass_heur_need = max(1, int(self.cec_seed_min_pass_heur))
        if int(a.probe_pass_heur) < pass_heur_need and int(af.probe_pass_heur) < pass_heur_need:
            return False, "no_pass_heur", meta

        stp = int(step)
        if int(self._sponsor_step) != stp:
            self._sponsor_step = stp
            self._sponsor_count_step = 0
        if int(self._sponsor_count_step) >= max(0, int(self.cec_trial_max_per_step)):
            return False, "step_trial_quota", meta
        if int(af.last_trial_step) > stp - max(0, int(self.cec_family_cooldown_steps)):
            return False, "family_cooldown", meta
        if int(a.trial_won) > 0:
            return False, "already_trial_won", meta
        if int(af.trial_seed_used) >= max(0, int(self.cec_direct_pass_seed_trials_per_family)):
            return False, "direct_pass_seed_budget_exhausted", meta

        self._sponsor_count_step += 1
        a.trial_sponsored += 1
        a.last_trial_step = stp
        a.last_trial_kind = "direct_pass_seed"
        af.trial_sponsored += 1
        af.last_trial_step = stp
        af.trial_seed_used += 1
        return True, "direct_pass_seed", meta

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

    def on_progress(self, used_calls: int, budget_total: int, best_total_seen: float, ctx: Optional[Dict[str, Any]] = None) -> None:
        """Advance horizon tickets and update AOS windows.

        NOTE: detailed_place should call this EVERY step; otherwise horizon credit never matures.
        """
        used_calls = int(used_calls)
        if used_calls <= int(getattr(self, "_last_progress_calls", -1)):
            return
        self._last_progress_calls = int(used_calls)

        cur_best = float(best_total_seen)
        ctx0 = ctx or {}
        stage = str(ctx0.get("stage", "") or "").strip().lower()
        if stage not in {"early", "mid", "late"}:
            try:
                stage = str(self._stage(self._budget_progress(used_calls, budget_total)))
            except Exception:
                stage = ""

        # Update realized baseline for both fine and coarse ctx keys
        try:
            ck_fine = str(ctx0.get("ctx_key", "") or "")
            ck_coarse = str(ctx0.get("release_ctx_key", "") or "")
            if ck_fine:
                self.observe_best_total(ctx_key=ck_fine, stage=stage, used_calls=used_calls, best_total_seen=cur_best)
            if ck_coarse:
                self.observe_best_total(ctx_key=ck_coarse, stage=stage, used_calls=used_calls, best_total_seen=cur_best)
        except Exception:
            pass

        if (not self.tickets) and (not self._aos_tickets):
            return

        # --- AOS tickets (short horizon) ---
        if bool(self.aos_enabled) and self._aos_tickets:
            kept_aos: List[_AosTicket] = []
            for tk in self._aos_tickets:
                try:
                    if float(cur_best) < float(getattr(tk, "min_best_total", 1.0e30)):
                        tk.min_best_total = float(cur_best)
                        tk.min_best_calls = int(used_calls)
                except Exception:
                    pass
                if used_calls < int(tk.expire_calls):
                    kept_aos.append(tk)
                    continue

                if str(self.aos_gain_mode) == "end":
                    gain_long = float(tk.start_best_total) - float(cur_best)
                    span_calls = max(1, int(tk.expire_calls) - int(tk.start_calls))
                else:
                    mb = float(getattr(tk, "min_best_total", cur_best))
                    mc = int(getattr(tk, "min_best_calls", -1))
                    gain_long = float(tk.start_best_total) - float(mb)
                    span_calls = max(1, (mc - int(tk.start_calls)) if mc > int(tk.start_calls) else (int(tk.expire_calls) - int(tk.start_calls)))
                if gain_long < 0.0:
                    gain_long = 0.0

                stg = str(tk.stage or "").strip().lower()
                ck = str(tk.ctx_key or "").strip()
                rate_best = float(self._atomic_rate(ck))
                rate_real = float(self._aos_realized_rate(ck, stg))
                if rate_real > 0.0:
                    rate0 = min(rate_best, float(self.aos_cf_cap_mult) * float(rate_real))
                else:
                    rate0 = min(rate_best, float(rate_real))
                rate0 = float(max(0.0, rate0))
                exp_atomic = float(rate0) * float(span_calls) * float(self.aos_cf_discount)
                credit = float(gain_long) - float(exp_atomic)

                fam = str(tk.family or "").strip()
                if stg in {"early", "mid", "late"} and fam:
                    w = self._get_win(self._aos_fam_win, (stg, fam), int(self.aos_window_calls))
                    w.add(now_calls=used_calls, gain=float(credit), calls=int(span_calls))
                    self._aos_stage_total_n[stg] = int(self._aos_stage_total_n.get(stg, 0)) + 1
            self._aos_tickets = kept_aos

        # --- Existing long-horizon tickets (CEC/release) ---
        kept: List[_Ticket] = []
        for tk in self.tickets:
            try:
                if float(cur_best) < float(getattr(tk, "min_best_total", 1.0e30)):
                    tk.min_best_total = float(cur_best)
                    tk.min_best_calls = int(used_calls)
            except Exception:
                pass
            if used_calls < int(tk.expire_calls):
                kept.append(tk)
                continue
            span = max(1, used_calls - int(tk.start_calls))
            gain_long = float(tk.start_best_total) - float(cur_best)
            # BC^2-CEC: counterfactual net gain
            gain_cf = float(gain_long) - float(getattr(tk, "expected_atomic_gain", 0.0))
            credit_gain = float(gain_cf) if bool(self.cec_counterfactual_credit) else float(gain_long)

            if credit_gain > 0.0:
                a = self.agg_by_src.get(str(tk.src))
                if a is not None:
                    al = float(self.alpha_long)
                    roi_long = float(credit_gain) / float(span)
                    a.roi_long = (1.0 - al) * float(a.roi_long) + al * float(roi_long)
            if str(tk.src) == "macro" and str(tk.ctx_key) and str(tk.family):
                ca = self._get_ctx_agg("macro", str(tk.ctx_key), str(tk.family))
                fa = self._get_family_agg("macro", str(tk.family))
                al = float(self.alpha_long)
                roi_long_ctx = float(credit_gain) / float(span)
                ca.roi_long = (1.0 - al) * float(ca.roi_long) + al * float(roi_long_ctx)
                fa.roi_long = (1.0 - al) * float(fa.roi_long) + al * float(roi_long_ctx)
                # audit
                ca.last_gain_long = float(gain_long)
                ca.last_gain_cf = float(gain_cf)
                ca.last_atomic_exp_gain = float(getattr(tk, "expected_atomic_gain", 0.0))
                if bool(tk.sponsored):
                    # Only promote in late stage; keep release strictly ctx-local.
                    _stage = str(tk.ctx_key or "").split("|", 1)[0] if str(tk.ctx_key or "") else ""
                    if _stage == "late":
                        release_ok, reason, _rmeta = self._soft_release_eval(
                            ctx_key=str(tk.ctx_key),
                            family=str(tk.family),
                            credit_gain=float(credit_gain),
                        )

                        if release_ok:
                            ca.released = 1
                            ca.last_release_step = int(used_calls)
                            ca.last_release_reason = str(reason)
                            # Upgrade: candidate -> released
                            if bool(self.cec_candidate_release):
                                ca.candidate = 0
                                ca.candidate_mature_failures = 0
                        else:
                            # v2.4: candidate grace
                            if bool(self.cec_candidate_release) and int(getattr(ca, "candidate", 0)) == 1:
                                ca.candidate_mature_failures = int(getattr(ca, "candidate_mature_failures", 0)) + 1
                                grace_calls = int(self._candidate_grace_calls(_stage))
                                age_calls = int(used_calls) - int(getattr(ca, "last_candidate_step", -10**9))
                                if (
                                    int(getattr(ca, "candidate_mature_failures", 0)) >= int(self.cec_candidate_grace_maturities)
                                    and int(age_calls) >= int(grace_calls)
                                ):
                                    ca.candidate = 0
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
        rel_any = False
        if comp == "macro" and not str(family or ""):
            rel_any = self.release_any("macro", ctx=(ctx or {"stage": stage, "ctx_key": ""}))
        _, _, ratio, slack, n = self._share_ratio(comp)
        if n >= int(self.share_min_samples):
            if float(ratio) < (1.0 - float(slack)):
                if not (comp == "macro" and stage == "late" and (float(self.roi_long("macro")) > 0.0 or rel or rel_any)):
                    st.allow_last = False
                    st.deny_last_reason = "share_low"
                    return False, "share_low"

        roi_floor = float(cfg.get("roi_floor", -1.0))
        cold_start_allow = bool(cfg.get("allow_cold_start", True))
        if float(roi) < float(roi_floor) and not (cold_start_allow and int(st.fired) <= 0):
            if not (comp == "macro" and stage == "late" and (float(self.roi_long("macro")) > 0.0 or rel or rel_any)):
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
        horizon = max(int(self.cec_ticket_horizon_min), int(float(self.cec_ticket_horizon_frac) * float(budget_total))) if budget_total > 0 else int(self.cec_ticket_horizon_min)

        # v2.3: remaining-budget-aware horizon for sponsored macro tickets.
        # We want the first sponsored macro win to mature before the run ends.
        if comp == "macro" and bool(sponsored) and budget_total > 0:
            remaining = max(0, int(budget_total) - int(used_calls))
            stage0 = str(ctx_key or "").split("|", 1)[0] if str(ctx_key or "") else ""
            if stage0 == "late":
                rem_h = max(int(self.cec_ticket_horizon_min), int(float(self.cec_ticket_horizon_late_remaining_frac) * float(remaining)))
                horizon = min(int(self.cec_ticket_horizon_late_cap), int(horizon), int(rem_h))
            elif stage0 == "mid":
                rem_h = max(int(self.cec_ticket_horizon_min), int(float(self.cec_ticket_horizon_mid_remaining_frac) * float(remaining)))
                horizon = min(int(self.cec_ticket_horizon_mid_cap), int(horizon), int(rem_h))
        # v2.6: In BC^2-CEC, treat late-stage accepted macro wins as sponsored by default.
        stage0 = str(ctx_key or "").split("|", 1)[0] if str(ctx_key or "") else ""
        if bool(self.cec_enabled) and bool(self.cec_counterfactual_credit) and comp == "macro" and stage0 == "late":
            sponsored = True

        # BC^2-CEC: improved counterfactual baseline snapshot at ticket creation.
        # Key fix: avoid over-penalizing macros when realized-rate is missing/noisy;
        # default fallback becomes atomic baseline (discounted), NOT global realized rate.
        ck0 = str(ctx_key or "")
        stg0 = self._cf_stage_from_ctx_key(ck0)
        d0 = float(self._cf_discount_for_stage(stg0))
        cap0 = float(self._cf_cap_for_stage(stg0))

        rate_best = float(self._atomic_rate(ck0))

        # Gate realized-rate by min_real_calls (EWMA calls).
        rate_real_ctx = 0.0
        try:
            ag = self.real_by_ctx.get(ck0)
            if ag is not None and float(ag.calls) >= float(self.cec_cf_min_real_calls) and float(ag.rate) > 0.0:
                rate_real_ctx = float(ag.rate)
        except Exception:
            rate_real_ctx = 0.0

        rate_real_global = 0.0
        try:
            if float(self.real_agg.calls) >= float(self.cec_cf_min_real_calls) and float(self.real_agg.rate) > 0.0:
                rate_real_global = float(self.real_agg.rate)
        except Exception:
            rate_real_global = 0.0

        mode = str(getattr(self, "cec_cf_mode", "atomic_cap_ctx_real"))
        if mode == "atomic":
            real_used = 0.0
            rate0 = float(rate_best)
        elif mode == "atomic_cap_real":
            real_used = float(rate_real_ctx) if rate_real_ctx > 0.0 else (float(rate_real_global) if rate_real_global > 0.0 else float(self.cec_cf_fallback_real_rate))
            if real_used > 0.0:
                rate0 = min(float(rate_best), float(cap0) * float(real_used))
            else:
                rate0 = float(rate_best)
        else:
            # recommended
            real_used = float(rate_real_ctx)
            if real_used > 0.0:
                rate0 = min(float(rate_best), float(cap0) * float(real_used))
            else:
                rate0 = float(rate_best)

        rate0 = float(max(0.0, rate0))
        exp_atomic_gain = float(rate0) * float(horizon) * float(d0)
        self.tickets.append(
            _Ticket(
                src=comp,
                start_calls=used_calls,
                start_best_total=best_total_seen,
                expire_calls=used_calls + horizon,
                ctx_key=str(ctx_key or ""),
                family=str(family or ""),
                sponsored=bool(sponsored),
                start_atomic_rate=float(rate0),
                start_real_rate=float(real_used),
                start_cf_discount=float(d0),
                start_cf_cap_mult=float(cap0),
                start_cf_mode=str(mode),
                expected_atomic_gain=float(exp_atomic_gain),
                min_best_total=float(best_total_seen),
                min_best_calls=int(used_calls),
            )
        )

        # v2.8 (AOS): create a short-horizon ticket for macro family credit updates.
        if bool(self.aos_enabled) and comp == "macro":
            stg = self._stage_from_ctx_key(str(ctx_key or ""))
            if stg not in {"early", "mid", "late"}:
                try:
                    stg = str(self._stage(self._budget_progress(used_calls, budget_total)))
                except Exception:
                    stg = ""
            fam = str(family or "").strip()
            ck = str(ctx_key or "").strip()
            if stg in {"early", "mid", "late"} and fam and ck and budget_total > 0:
                rem = max(0, int(budget_total) - int(used_calls))
                h = max(int(self.aos_horizon_min), int(float(self.aos_horizon_frac) * float(budget_total)))
                if stg == "late":
                    h = min(int(self.aos_horizon_late_cap), int(h), int(max(self.aos_horizon_min, rem)))
                elif stg == "mid":
                    h = min(int(self.aos_horizon_mid_cap), int(h), int(max(self.aos_horizon_min, rem)))
                else:
                    h = min(int(h), int(max(self.aos_horizon_min, rem)))
                self._aos_tickets.append(
                    _AosTicket(
                        start_calls=int(used_calls),
                        start_best_total=float(best_total_seen),
                        expire_calls=int(used_calls) + int(h),
                        stage=str(stg),
                        ctx_key=str(ck),
                        family=str(fam),
                        min_best_total=float(best_total_seen),
                        min_best_calls=int(used_calls),
                    )
                )
        if comp == "macro" and str(ctx_key or "") and str(family or "") and bool(sponsored):
            ca = self._get_ctx_agg("macro", str(ctx_key), str(family))
            ca.trial_won += 1
            if bool(self.cec_candidate_release):
                ca.candidate = 1
                ca.last_candidate_step = int(used_calls)
                ca.candidate_mature_failures = 0
            self._get_family_agg("macro", str(family)).trial_won += 1

    def roi_long(self, comp: str) -> float:
        ag = self.agg_by_src.get(str(comp))
        return float(ag.roi_long) if ag is not None else 0.0

    def release_active(self, comp: str, family: str = "", ctx: Optional[Dict[str, Any]] = None) -> bool:
        if not bool(self.cec_enabled):
            return False
        if str(comp) != "macro":
            return False
        ctx0 = ctx or {}
        if str(ctx0.get("stage", "")) != "late":
            return False
        ctx_key = self._active_ctx_key(ctx0)
        fam = str(family or "")
        a = self._get_ctx_agg("macro", ctx_key, fam)
        fa = self._get_family_agg("macro", fam)
        ts = self._trial_score("macro", ctx_key, fam)

        if bool(a.released):
            if (
                float(a.roi_long) <= float(self.cec_release_keep_roi_floor_local)
                and float(fa.roi_long) <= float(self.cec_release_keep_roi_floor_family)
                and float(ts.get("trial_score", 0.0)) <= float(self.cec_release_keep_trial_score_floor)
            ):
                a.released = 0
            else:
                a.release_hits += 1
                return True

        release_ok, reason, _meta = self._soft_release_eval(
            ctx_key=ctx_key,
            family=fam,
            credit_gain=None,
        )
        if release_ok:
            a.released = 1
            a.last_release_reason = str(reason)
            if bool(self.cec_candidate_release):
                a.candidate = 0
                a.candidate_mature_failures = 0
            a.release_hits += 1
            return True
        return False

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

        # v2.5 precision-first:
        # candidate is only a light bias (family ordering), not a strong source-level quota bias.
        # Only released families can increase macro quota.
        if comp == "macro" and not str(family or ""):
            if self.release_any("macro", ctx=(ctx or {"stage": stage, "ctx_key": ""})):
                q = min(qmax, q + 1)
        else:
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
            "cec_v2_enabled": int(bool(self.cec_enabled)),
            "heur_rate_ewma": float(self.heur_agg.rate),
            "cec_credit_metric": str("gain_cf" if bool(self.cec_counterfactual_credit) else "gain_long"),
            "aos_enabled": int(bool(getattr(self, "aos_enabled", False))),
        }
        if bool(self.cec_counterfactual_credit):
            out["cec_cf_discount"] = float(getattr(self, "cec_cf_discount", 1.0))
            out["cec_cf_discount_by_stage"] = dict(getattr(self, "cec_cf_discount_by_stage", {}) or {})
            out["cec_cf_cap_mult"] = float(getattr(self, "cec_cf_cap_mult", 2.0))
            out["cec_cf_cap_mult_by_stage"] = dict(getattr(self, "cec_cf_cap_mult_by_stage", {}) or {})
            out["cec_cf_mode"] = str(getattr(self, "cec_cf_mode", ""))
            out["cec_cf_min_real_calls"] = int(getattr(self, "cec_cf_min_real_calls", 0))
            out["cec_cf_fallback_real_rate"] = float(getattr(self, "cec_cf_fallback_real_rate", 0.0))
        if bool(self.cec_counterfactual_credit):
            try:
                out["atomic_rate_ctx_n"] = int(len(self.atomic_by_ctx))
                # cap to keep logs bounded
                out["atomic_rate_by_ctx"] = {k: float(v.rate) for k, v in list(self.atomic_by_ctx.items())[:32]}
            except Exception:
                pass
        # realized-rate audit (debug counterfactual scale)
        try:
            out["realized_rate_ewma"] = float(self.real_agg.rate)
            out["realized_rate_ctx_n"] = int(len(self.real_by_ctx))
            out["realized_rate_by_ctx"] = {k: float(v.rate) for k, v in list(self.real_by_ctx.items())[:32]}
        except Exception:
            pass
        cec_ctx: Dict[str, Dict[str, Any]] = {}
        cec_family: Dict[str, Dict[str, Any]] = {}
        rel_total = 0
        cand_total = 0
        probe_total = 0
        seed_total = 0
        for (comp, ctx_key, family), a in self.ctx_agg.items():
            ts = self._trial_score(comp, ctx_key, family)
            probe_total += int(a.probe_n)
            rel_total += int(a.released)
            cand_total += int(getattr(a, "candidate", 0))
            k = f"{comp}|{ctx_key}|{family}"
            cec_ctx[k] = {
                "probe_n": int(a.probe_n),
                "probe_pass_heur": int(a.probe_pass_heur),
                "probe_pass_cur": int(a.probe_pass_cur),
                "probe_margin_heur": float(a.probe_margin_heur),
                "probe_margin_cur": float(a.probe_margin_cur),
                "probe_calls": float(a.probe_calls),
                "probe_utility": float(self._probe_utility(comp, ctx_key, family)),
                "probe_edge_local": float(ts.get("edge_local", 0.0)),
                "probe_edge_family": float(ts.get("edge_family", 0.0)),
                "trial_score": float(ts.get("trial_score", 0.0)),
                "lambda_local": float(ts.get("lambda_local", 0.0)),
                "trial_sponsored": int(a.trial_sponsored),
                "trial_won": int(a.trial_won),
                "roi_long": float(a.roi_long),
                "last_gain_long": float(getattr(a, "last_gain_long", 0.0)),
                "last_gain_cf": float(getattr(a, "last_gain_cf", 0.0)),
                "last_atomic_exp_gain": float(getattr(a, "last_atomic_exp_gain", 0.0)),
                "released": int(a.released),
                "candidate": int(getattr(a, "candidate", 0)),
                "candidate_hits": int(getattr(a, "candidate_hits", 0)),
                "last_candidate_step": int(getattr(a, "last_candidate_step", -10**9)),
                "release_hits": int(a.release_hits),
                "last_trial_kind": str(getattr(a, "last_trial_kind", "") or ""),
                "last_release_reason": str(getattr(a, "last_release_reason", "") or ""),
                "prior_src": str(ts.get("prior_src", "") or ""),
            }
        for (comp, family), af in self.family_agg.items():
            seed_total += int(af.trial_seed_used)
            k = f"{comp}|{family}"
            cec_family[k] = {
                "probe_n": int(af.probe_n),
                "probe_pass_heur": int(af.probe_pass_heur),
                "probe_pass_cur": int(af.probe_pass_cur),
                "probe_margin_heur": float(af.probe_margin_heur),
                "probe_margin_cur": float(af.probe_margin_cur),
                "probe_calls": float(af.probe_calls),
                "probe_edge": float(self._probe_edge_family(comp, family)),
                "trial_seed_used": int(af.trial_seed_used),
                "trial_sponsored": int(af.trial_sponsored),
                "trial_won": int(af.trial_won),
                "roi_long": float(af.roi_long),
                "last_trial_step": int(af.last_trial_step),
            }
        out["cec_probe_total"] = int(probe_total)
        out["cec_release_total"] = int(rel_total)
        out["cec_candidate_total"] = int(cand_total)
        # v2.1: audit the active rule
        out["cec_release_rule"] = "gain_cf_pos_or_score_gate" if bool(self.cec_counterfactual_credit) else "gain_long_pos_or_score_gate"
        out["cec_family_total"] = int(len(self.family_agg))
        out["cec_seed_total"] = int(seed_total)
        out["cec_ctx"] = cec_ctx
        out["cec_family"] = cec_family

        # AOS audit (stage-conditioned sliding window credit)
        if bool(getattr(self, "aos_enabled", False)):
            try:
                out["aos_window_calls"] = int(getattr(self, "aos_window_calls", 0))
                out["aos_gain_mode"] = str(getattr(self, "aos_gain_mode", ""))
                out["aos_ucb_c"] = float(getattr(self, "aos_ucb_c", 0.0))
                out["aos_cf_discount"] = float(getattr(self, "aos_cf_discount", 0.0))
                out["aos_cf_cap_mult"] = float(getattr(self, "aos_cf_cap_mult", 0.0))
                out["aos_stage_total_n"] = {k: int(v) for k, v in (getattr(self, "_aos_stage_total_n", {}) or {}).items()}
                _pairs = list((getattr(self, "_aos_fam_win", {}) or {}).items())[:48]
                out["aos_family_rate"] = {f"{k0}|{k1}": float(v.rate) for (k0, k1), v in _pairs}
                out["aos_family_n"] = {f"{k0}|{k1}": int(v.n) for (k0, k1), v in _pairs}
            except Exception:
                pass

        # v2.1: stage-family aggregates (auditability for stage-aware priors)
        cec_family_stage: Dict[str, Dict[str, Any]] = {}
        for (comp, stage, family), af in self.family_stage_agg.items():
            k = f"{comp}|{stage}|{family}"
            cec_family_stage[k] = {
                "probe_n": int(af.probe_n),
                "probe_pass_heur": int(af.probe_pass_heur),
                "probe_pass_cur": int(af.probe_pass_cur),
                "probe_margin_heur": float(af.probe_margin_heur),
                "probe_margin_cur": float(af.probe_margin_cur),
                "probe_calls": float(af.probe_calls),
                "probe_edge": float(self._probe_edge_family_stage(comp, stage, family)) if int(af.probe_n) > 0 else 0.0,
                "trial_sponsored": int(af.trial_sponsored),
                "trial_won": int(af.trial_won),
                "roi_long": float(af.roi_long),
                "last_trial_step": int(af.last_trial_step),
            }
        out["cec_family_stage"] = cec_family_stage

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
