"""A stronger, budget-aware macro engine.

Old macro = build_candidate_pool + many evaluator.evaluate() calls per macro step.
Under eval-call budgets, that becomes a fixed tax and often hurts.

This engine:
  - uses analytic O(S) deltas to generate candidates (no eval-call)
  - verifies only top-k candidates with real evaluator (1~2 eval-calls/step)
  - ALNS-style operator adaptation (weight + cooldown) to suppress bad macros
  - monotone by default (no non-improving macro steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import numpy as np

from layout.delta_eval import ObjectiveParams, estimate_action_delta

class EvalBudgetExceeded(RuntimeError):
    """Raised by capped evaluate_assign wrappers when probe eval-call budget is exhausted."""
    pass


def _rng_randint(rng, low: int, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(low, high))
    raise TypeError(f"Unsupported RNG type: {type(rng)}")


def _nearest_sites(sites_xy: np.ndarray, anchor_xy: np.ndarray, cand_sites: np.ndarray, k: int) -> List[int]:
    if cand_sites.size == 0:
        return []
    pts = sites_xy[cand_sites]
    d2 = np.sum((pts - anchor_xy[None, :]) ** 2, axis=1)
    idx = np.argsort(d2)[: max(1, min(int(k), d2.shape[0]))]
    return [int(cand_sites[t]) for t in idx.tolist()]


def _farthest_sites(sites_xy: np.ndarray, anchor_xy: np.ndarray, cand_sites: np.ndarray, k: int) -> List[int]:
    if cand_sites.size == 0:
        return []
    pts = sites_xy[cand_sites]
    d2 = np.sum((pts - anchor_xy[None, :]) ** 2, axis=1)
    idx = np.argsort(-d2)[: max(1, min(int(k), d2.shape[0]))]
    return [int(cand_sites[t]) for t in idx.tolist()]


def _apply_swap(assign: np.ndarray, i: int, j: int) -> np.ndarray:
    a = assign.copy()
    if i == j:
        return a
    a[i], a[j] = int(a[j]), int(a[i])
    return a


def _apply_relocate_perm(assign: np.ndarray, i: int, site_id: int) -> np.ndarray:
    """Permutation-safe relocate (swap with occupant if needed)."""
    a = assign.copy()
    i = int(i)
    site_id = int(site_id)
    if i < 0 or i >= a.shape[0]:
        return a
    occ = np.where(a == site_id)[0]
    j = int(occ[0]) if occ.size > 0 else -1
    if j >= 0 and j != i:
        a[i], a[j] = int(a[j]), int(a[i])
        return a
    a[i] = site_id
    return a


def _dedup_ints(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _pairwise_comm_cost(pos_xy: np.ndarray, chips: List[int], traffic_sym: np.ndarray) -> float:
    """Weighted pairwise squared-distance cost inside a chip set.

    Used to make block_relocate structurally different from ruin: block should explicitly
    improve internal communication geometry for a small correlated group.
    """
    if not chips or len(chips) <= 1:
        return 0.0
    idx = [int(i) for i in chips]
    pts = np.asarray(pos_xy, dtype=np.float64)[np.asarray(idx, dtype=int)]
    tr = np.asarray(traffic_sym, dtype=np.float64)
    tot = 0.0
    for a in range(len(idx)):
        ia = int(idx[a])
        for b in range(a + 1, len(idx)):
            ib = int(idx[b])
            w = float(tr[ia, ib] + tr[ib, ia])
            if w <= 0.0:
                continue
            d2 = float(np.sum((pts[a] - pts[b]) ** 2))
            tot += w * d2
    return float(tot)


@dataclass
class OpStat:
    tries: int = 0
    success: int = 0
    fail: int = 0
    ewma_gain_per_call: float = 0.0
    ewma_gain: float = 0.0
    ewma_calls: float = 0.0
    # Counterfactual probe EWMA (operator gain-per-call minus atomic baseline rate)
    ewma_cf_gain_per_call: float = 0.0
    ewma_cf_gain: float = 0.0
    ewma_cf_calls: float = 0.0
    # Probe accounting
    probe_n: int = 0
    probe_last_stage: str = ""
    cooldown: int = 0
    weight: float = 1.0


@dataclass
class EliteEntry:
    sig: str
    total: float
    comm: float
    therm: float
    assign: np.ndarray
    eval: Dict[str, Any]


def _hamming_assign(a: np.ndarray, b: np.ndarray) -> int:
    try:
        return int(np.count_nonzero(np.asarray(a, dtype=int) != np.asarray(b, dtype=int)))
    except Exception:
        return 0


class MacroEngine:
    def __init__(self, macro_cfg: Dict[str, Any], obj: ObjectiveParams, rng: Any) -> None:
        self.cfg = macro_cfg or {}
        self.obj = obj
        self.rng = rng
        self.stats: Dict[str, OpStat] = {}

        adapt_cfg = (self.cfg.get("adapt") or {}) if isinstance(self.cfg, dict) else {}
        self.adapt_enabled = bool(adapt_cfg.get("enabled", True))
        self.alpha = float(adapt_cfg.get("ewma_alpha", 0.2))
        self.fail_cooldown = int(adapt_cfg.get("fail_cooldown", 20))
        self.success_cooldown = int(adapt_cfg.get("success_cooldown", 0))
        self.weight_floor = float(adapt_cfg.get("weight_floor", 0.1))
        self.weight_cap = float(adapt_cfg.get("weight_cap", 5.0))
        self.success_boost = float(adapt_cfg.get("success_boost", 0.2))
        self.fail_penalty = float(adapt_cfg.get("fail_penalty", 0.2))

        delta_cfg = (self.cfg.get("delta") or {}) if isinstance(self.cfg, dict) else {}
        self.cand_k = int(delta_cfg.get("candidate_k", 30))
        self.eval_topk = int(delta_cfg.get("eval_topk_per_step", 2))
        self.partner_topk = int(delta_cfg.get("partner_topk", 8))
        self.empty_site_k = int(delta_cfg.get("empty_site_k", 20))
        self.hot_slots_k = int(delta_cfg.get("hot_slots_k", 12))
        self.therm_hot_k = int(delta_cfg.get("therm_hot_k", 10))
        self.therm_cold_k = int(delta_cfg.get("therm_cold_k", 10))
        self.monotone = bool(delta_cfg.get("monotone", True))

        # New macro families: chain and ruin_repair
        chain_cfg = (self.cfg.get("chain") or {}) if isinstance(self.cfg, dict) else {}
        self.chain_len = int(chain_cfg.get("chain_len", 5))
        self.chain_propose = int(chain_cfg.get("propose_chains", 24))
        self.chain_eval_topk = int(chain_cfg.get("eval_topk", 6))
        self.chain_step_topk = int(chain_cfg.get("step_topk", 10))
        self.chain_step_sample_topk = int(chain_cfg.get("step_sample_topk", 3))

        rr_cfg = (self.cfg.get("ruin_repair") or {}) if isinstance(self.cfg, dict) else {}
        self.rr_ratios = [float(x) for x in (rr_cfg.get("ruin_ratios", [0.10, 0.20, 0.30]) or [])]
        self.rr_candidates_per_ratio = int(rr_cfg.get("candidates_per_ratio", 2))
        self.rr_eval_topk = int(rr_cfg.get("eval_topk", 6))
        self.rr_repair_steps = int(rr_cfg.get("repair_steps", 6))
        self.rr_repair_step_topk = int(rr_cfg.get("repair_step_topk", 12))
        self.rr_repair_step_sample_topk = int(rr_cfg.get("repair_step_sample_topk", 3))

        # v2.2: strengthen ruin-repair (otherwise it often never produces a net-improving candidate)
        # - target ruined chips to "hard" parts (traffic-hot / thermal-hot)
        # - make ruin move bigger (farthest-empty relocate)
        # - rank candidates by analytic delta sum instead of random shuffling
        self.rr_hot_comm_k = int(rr_cfg.get("hot_comm_k", 12))
        self.rr_hot_therm_k = int(rr_cfg.get("hot_therm_k", 10))
        self.rr_hot_prob = float(rr_cfg.get("hot_prob", 0.75))
        self.rr_ruin_site_mode = str(rr_cfg.get("ruin_site_mode", "far"))  # far | random
        self.rr_far_site_k = int(rr_cfg.get("far_site_k", 24))
        self.rr_eval_random = int(rr_cfg.get("eval_random", 1))
        self.rr_repair_pick = str(rr_cfg.get("repair_pick", "best"))  # best | sample

        # New macro family: block_relocate
        br_cfg = (self.cfg.get("block_relocate") or {}) if isinstance(self.cfg, dict) else {}
        self.br_block_size = int(br_cfg.get("block_size", 3))
        self.br_seed_topk = int(br_cfg.get("seed_topk", 10))
        self.br_partner_topk = int(br_cfg.get("partner_topk", 8))
        self.br_external_partner_topk = int(br_cfg.get("external_partner_topk", 4))
        self.br_anchor_topk = int(br_cfg.get("anchor_topk", 6))
        self.br_candidate_blocks = int(br_cfg.get("candidate_blocks", 18))
        self.br_eval_topk = int(br_cfg.get("eval_topk", 6))
        self.br_use_empty_only = bool(br_cfg.get("use_empty_only", True))

        # v2.2: strengthen block-relocate to be objective-aware and distinct from ruin
        self.br_perm_sites_k = int(br_cfg.get("perm_sites_k", 6))
        self.br_perm_limit = int(br_cfg.get("perm_limit", 120))
        self.br_score_w_est = float(br_cfg.get("score_w_est", 1.0))
        self.br_score_w_internal = float(br_cfg.get("score_w_internal", 1e-6))
        self.br_score_w_compact = float(br_cfg.get("score_w_compact", 1e-6))
        self.br_score_w_centroid = float(br_cfg.get("score_w_centroid", 5e-7))

        # ------------------------------------------------------------------
        # Elite archive (for relinking / robust escape)
        # ------------------------------------------------------------------
        ea_cfg = (self.cfg.get("elite_archive") or {}) if isinstance(self.cfg, dict) else {}
        self.elite_enabled = bool(ea_cfg.get("enabled", True))
        self.elite_max_size = int(ea_cfg.get("max_size", 8))
        self.elite_min_hamming_frac = float(ea_cfg.get("min_hamming_frac", 0.08))
        self.elite_keep_top_total = int(ea_cfg.get("keep_top_total", self.elite_max_size))
        self._elite: List[EliteEntry] = []
        self._elite_sig_to_idx: Dict[str, int] = {}

        # ------------------------------------------------------------------
        # New enhancement components as macro families:
        #   - relink (elite archive + path relinking)
        #   - shake  (ILS-style perturbation + short repair)
        #   - tabu_search (tabu local search with controlled non-improving moves)
        # ------------------------------------------------------------------
        rl_cfg = (self.cfg.get("relink") or self.cfg.get("elite_relink") or {}) if isinstance(self.cfg, dict) else {}
        self.rlk_eval_topk = int(rl_cfg.get("eval_topk", 12))
        self.rlk_pick_k = int(rl_cfg.get("pick_k", 24))
        self.rlk_moves_per_step = int(rl_cfg.get("moves_per_step", 18))
        self.rlk_allow_worsen = bool(rl_cfg.get("allow_worsen", True))

        sh_cfg = (self.cfg.get("shake") or self.cfg.get("ils_shake") or {}) if isinstance(self.cfg, dict) else {}
        self.shake_candidates = int(sh_cfg.get("candidates", 16))
        self.shake_kick_k = int(sh_cfg.get("kick_k", 8))
        self.shake_kick_rounds = int(sh_cfg.get("kick_rounds", 2))
        self.shake_eval_topk = int(sh_cfg.get("eval_topk", 12))
        self.shake_repair_steps = int(sh_cfg.get("repair_steps", 4))
        self.shake_repair_eval_topk = int(sh_cfg.get("repair_eval_topk", 4))
        self.shake_allow_worsen = bool(sh_cfg.get("allow_worsen", True))

        tb_cfg = (self.cfg.get("tabu_search") or self.cfg.get("tabu") or {}) if isinstance(self.cfg, dict) else {}
        self.tabu_steps = int(tb_cfg.get("steps", 12))
        self.tabu_tenure = int(tb_cfg.get("tenure", 8))
        self.tabu_cand_k = int(tb_cfg.get("cand_k", 80))
        self.tabu_eval_topk = int(tb_cfg.get("eval_topk", 6))
        self.tabu_allow_worsen = bool(tb_cfg.get("allow_worsen", True))
        self.tabu_aspire_eps = float(tb_cfg.get("aspire_eps", 1e-4))

    def _stat(self, name: str) -> OpStat:
        if name not in self.stats:
            self.stats[name] = OpStat()
        return self.stats[name]

    def tick(self) -> None:
        for st in self.stats.values():
            if st.cooldown > 0:
                st.cooldown -= 1

    def available(self, name: str) -> bool:
        return self._stat(name).cooldown <= 0

    def rank_macros(self, macro_names: List[str]) -> List[str]:
        scored = []
        for n in macro_names:
            st = self._stat(n)
            w = float(st.weight)
            if st.cooldown > 0:
                w *= 0.01
            scored.append((w, n))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [n for _, n in scored]

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, st in self.stats.items():
            out[str(k)] = {
                "tries": int(st.tries),
                "success": int(st.success),
                "fail": int(st.fail),
                "ewma_gain_per_call": float(st.ewma_gain_per_call),
                "ewma_cf_gain_per_call": float(getattr(st, "ewma_cf_gain_per_call", 0.0)),
                "probe_n": int(getattr(st, "probe_n", 0)),
                "probe_last_stage": str(getattr(st, "probe_last_stage", "")),
                "cooldown": int(st.cooldown),
                "weight": float(st.weight),
            }
        return out

    # ----------------------------
    # Stage-stratified probing feedback (operator-agnostic)
    # ----------------------------
    def apply_probe_feedback(
        self,
        name: str,
        raw_gain: float,
        raw_calls: float,
        atomic_gain: float,
        atomic_calls: float,
        stage: str = "",
        now_calls: int = 0,
        probe_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update per-operator probe statistics and (optionally) adaptation weights.

        This method is intended to be called by external stage-probe schedulers.
        It does not call the evaluator; it only consumes observed probe outcomes.
        """
        st = self._stat(str(name))
        cfg = probe_cfg or {}

        raw_gain = float(raw_gain)
        raw_calls = float(max(1.0, raw_calls))
        atomic_gain = float(atomic_gain)
        atomic_calls = float(max(1.0, atomic_calls))

        rate_op = float(raw_gain) / float(max(1e-9, raw_calls))
        rate_atomic = float(atomic_gain) / float(max(1e-9, atomic_calls))

        # Stage-wise discount (optional): early/mid/late can differ to avoid over-subtraction.
        stg = str(stage or "").strip().lower()
        d_by_stage = cfg.get("cf_discount_by_stage", {}) if isinstance(cfg.get("cf_discount_by_stage", {}), dict) else {}
        if isinstance(d_by_stage, dict) and stg in d_by_stage:
            cf_discount = float(d_by_stage.get(stg, 1.0))
        else:
            cf_discount = float(cfg.get("cf_discount", cfg.get("counterfactual_discount", 1.0)))
        if not (cf_discount == cf_discount):
            cf_discount = 1.0
        cf_discount = float(max(0.0, min(1.0, cf_discount)))

        # Confidence gate + cap to reduce noisy/over-strong baselines dominating credit.
        min_atomic_calls = int(cfg.get("min_atomic_calls_for_cf", 0))
        use_cf = bool(min_atomic_calls <= 0 or float(atomic_calls) >= float(min_atomic_calls))

        cf_cap_mult = float(cfg.get("cf_cap_mult", cfg.get("counterfactual_cap_mult", 1.5)))
        if not (cf_cap_mult == cf_cap_mult):
            cf_cap_mult = 1.5
        cf_cap_mult = float(max(0.5, min(5.0, cf_cap_mult)))

        if use_cf:
            rate_atomic_eff = min(float(rate_atomic), float(cf_cap_mult) * max(float(rate_op), 1e-12))
        else:
            rate_atomic_eff = 0.0

        rate_cf = float(rate_op) - float(cf_discount) * float(rate_atomic_eff)
        cf_gain = float(rate_cf) * float(raw_calls)

        a = float(cfg.get("ewma_alpha", getattr(self, "alpha", 0.2)))
        a = float(max(0.01, min(0.95, a)))
        # Update raw EWMA too (stabilizes weight updates when CF is noisy/disabled).
        st.ewma_gain = (1.0 - a) * float(getattr(st, "ewma_gain", 0.0)) + a * float(raw_gain)
        st.ewma_calls = (1.0 - a) * float(getattr(st, "ewma_calls", 0.0)) + a * float(raw_calls)
        st.ewma_gain_per_call = float(st.ewma_gain) / float(max(1e-9, st.ewma_calls))

        st.ewma_cf_gain = (1.0 - a) * float(getattr(st, "ewma_cf_gain", 0.0)) + a * float(cf_gain)
        st.ewma_cf_calls = (1.0 - a) * float(getattr(st, "ewma_cf_calls", 0.0)) + a * float(raw_calls)
        st.ewma_cf_gain_per_call = float(st.ewma_cf_gain) / float(max(1e-9, st.ewma_cf_calls))
        st.probe_n = int(getattr(st, "probe_n", 0)) + 1
        st.probe_last_stage = str(stage or "")

        update_weight = bool(cfg.get("update_weight", True))
        weight_metric = str(cfg.get("weight_metric", "cf")).lower()
        min_gain_per_call = float(cfg.get("min_gain_per_call", 0.0))
        use_ewma = bool(cfg.get("weight_use_ewma", True))
        if weight_metric in {"cf", "gain_cf", "counterfactual"}:
            metric = float(st.ewma_cf_gain_per_call) if use_ewma else float(rate_cf)
        else:
            metric = float(st.ewma_gain_per_call) if use_ewma else float(rate_op)

        boost_scale = float(cfg.get("success_boost_scale", 0.5))
        pen_scale = float(cfg.get("fail_penalty_scale", 0.5))
        boost_scale = float(max(0.0, min(2.0, boost_scale)))
        pen_scale = float(max(0.0, min(2.0, pen_scale)))

        if update_weight and bool(getattr(self, "adapt_enabled", True)):
            if metric > float(min_gain_per_call):
                st.weight = min(self.weight_cap, float(st.weight) * (1.0 + float(self.success_boost) * boost_scale))
                st.cooldown = max(int(st.cooldown), int(self.success_cooldown))
            else:
                st.weight = max(self.weight_floor, float(st.weight) * (1.0 - float(self.fail_penalty) * pen_scale))
                st.cooldown = max(int(st.cooldown), int(self.fail_cooldown))

        return {
            "name": str(name),
            "stage": str(stage or ""),
            "now_calls": int(now_calls),
            "raw_gain": float(raw_gain),
            "raw_calls": float(raw_calls),
            "atomic_gain": float(atomic_gain),
            "atomic_calls": float(atomic_calls),
            "rate_op": float(rate_op),
            "rate_atomic": float(rate_atomic),
            "rate_atomic_eff": float(rate_atomic_eff),
            "rate_cf": float(rate_cf),
            "cf_discount": float(cf_discount),
            "cf_cap_mult": float(cf_cap_mult),
            "use_cf": int(bool(use_cf)),
            "metric_used": float(metric),
            "weight": float(st.weight),
        }

    # ----------------------------
    # Elite archive (for relinking)
    # ----------------------------
    def observe_elite(self, assign: np.ndarray, eo: Dict[str, Any]) -> None:
        """Insert a solution into the elite archive (dedup + size cap + light diversity)."""
        if not bool(getattr(self, "elite_enabled", True)):
            return
        try:
            a = np.asarray(assign, dtype=int).reshape(-1)
        except Exception:
            return
        try:
            total = float((eo or {}).get("total_scalar", 1e30))
            comm = float((eo or {}).get("comm_norm", 0.0))
            therm = float((eo or {}).get("therm_norm", 0.0))
        except Exception:
            return

        # signature (content-based)
        try:
            md = hashlib.md5(a.tobytes()).hexdigest()
        except Exception:
            md = str(int(total * 1e9))

        # update existing
        idx = self._elite_sig_to_idx.get(md)
        if idx is not None and 0 <= int(idx) < len(self._elite):
            if total + 1e-12 < float(self._elite[int(idx)].total):
                self._elite[int(idx)] = EliteEntry(sig=md, total=total, comm=comm, therm=therm, assign=a.copy(), eval=dict(eo or {}))
            return

        # diversity gate: avoid near-duplicates unless very good
        min_h = int(max(1, float(self.elite_min_hamming_frac) * float(a.shape[0])))
        for e in self._elite:
            if _hamming_assign(a, e.assign) < min_h and total >= float(e.total) - 1e-9:
                return

        self._elite.append(EliteEntry(sig=md, total=total, comm=comm, therm=therm, assign=a.copy(), eval=dict(eo or {})))

        # keep a small archive: primarily by total
        self._elite.sort(key=lambda x: float(x.total))
        keep_n = int(max(2, min(int(self.elite_keep_top_total), int(self.elite_max_size))))
        self._elite = self._elite[:keep_n]
        self._elite_sig_to_idx = {e.sig: i for i, e in enumerate(self._elite)}

    def _pick_elite_target(self, cur_assign: np.ndarray) -> Optional[EliteEntry]:
        if len(self._elite) < 2:
            return None
        a = np.asarray(cur_assign, dtype=int).reshape(-1)
        # pick good-but-different target: among top half, maximize hamming distance
        cand = self._elite[: max(2, len(self._elite) // 2)]
        best = None
        best_d = -1
        for e in cand:
            d = _hamming_assign(a, e.assign)
            if d <= 0:
                continue
            if d > best_d:
                best_d = d
                best = e
        if best is None:
            for e in self._elite:
                d = _hamming_assign(a, e.assign)
                if d > best_d:
                    best_d = d
                    best = e
        return best

    def _empty_sites(self, assign: np.ndarray, Ns: int) -> np.ndarray:
        used = np.zeros(Ns, dtype=bool)
        used[np.asarray(assign, dtype=int)] = True
        return np.nonzero(~used)[0].astype(int)

    def _apply_act(self, assign: np.ndarray, act: Dict[str, Any]) -> np.ndarray:
        op = str(act.get("op", "none"))
        if op == "swap":
            return _apply_swap(assign, int(act.get("i", 0)), int(act.get("j", 0)))
        if op == "relocate":
            return _apply_relocate_perm(assign, int(act.get("i", 0)), int(act.get("site_id", 0)))
        return assign.copy()

    def _update_stat_end(self, st: OpStat, gain: float, calls: float, info: Dict[str, Any]) -> None:
        gain = float(max(0.0, gain))
        calls = float(max(1.0, calls))
        a = float(self.alpha)
        st.ewma_gain = (1.0 - a) * float(st.ewma_gain) + a * float(gain)
        st.ewma_calls = (1.0 - a) * float(st.ewma_calls) + a * float(calls)
        st.ewma_gain_per_call = float(st.ewma_gain) / float(max(1e-9, st.ewma_calls))

        if gain > 0.0:
            st.success += 1
            info["success"] = int(info.get("success", 0)) + 1
            if self.adapt_enabled:
                st.cooldown = max(st.cooldown, int(self.success_cooldown))
                st.weight = min(self.weight_cap, float(st.weight) * (1.0 + self.success_boost))
        else:
            st.fail += 1
            info["fail"] = int(info.get("fail", 0)) + 1
            if self.adapt_enabled:
                st.cooldown = max(st.cooldown, int(self.fail_cooldown))
                st.weight = max(self.weight_floor, float(st.weight) * (1.0 - self.fail_penalty))

    def _propose_chain_primitives(
        self,
        assign: np.ndarray,
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        a0 = []
        try:
            a0.extend(self.propose_actions("comm", assign, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
        except Exception:
            pass
        try:
            a0.extend(self.propose_actions("therm", assign, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
        except Exception:
            pass
        try:
            a0.extend(self.propose_actions("escape", assign, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
        except Exception:
            pass

        seen = set()
        out = []
        for act in a0:
            op = str(act.get("op"))
            if op == "swap":
                i = int(act.get("i", -1)); j = int(act.get("j", -1))
                a, b = (i, j) if i <= j else (j, i)
                key = ("swap", a, b)
            elif op == "relocate":
                key = ("relocate", int(act.get("i", -1)), int(act.get("site_id", -1)))
            else:
                key = (op,)
            if key in seen:
                continue
            seen.add(key)
            out.append(act)
            if len(out) >= max(20, int(self.cand_k)):
                break
        return out

    def _propose_block_relocate_candidates(
        self,
        assign: np.ndarray,
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
    ) -> List[Tuple[float, List[Dict[str, Any]], np.ndarray]]:
        _ = site_to_region
        a = np.asarray(assign, dtype=int)
        S = int(a.shape[0])
        Ns = int(sites_xy_mm.shape[0])
        pos = np.asarray(sites_xy_mm, dtype=np.float64)[a]
        all_sites = np.arange(Ns, dtype=int)
        empty = self._empty_sites(a, Ns)

        # seed chips: traffic-hot + thermal-hot
        seeds_comm = self._hot_slots_by_traffic(traffic_sym, max(4, int(self.br_seed_topk)))
        tdp = np.asarray(chip_tdp_w, dtype=np.float64)
        seeds_therm = [
            int(x)
            for x in np.argsort(-tdp)[: max(1, min(S, int(max(4, self.br_seed_topk // 2))))].tolist()
        ]
        seeds = _dedup_ints(seeds_comm + seeds_therm)

        candidates: List[Tuple[float, List[Dict[str, Any]], np.ndarray]] = []
        if not seeds:
            return candidates

        for seed in seeds[: max(1, int(self.br_seed_topk))]:
            # Build a correlated block around the seed using strong communication partners.
            partners = self._top_partners(traffic_sym, int(seed), max(2, int(self.br_partner_topk)))
            block = [int(seed)]
            for j in partners:
                if int(j) not in block:
                    block.append(int(j))
                if len(block) >= max(2, int(self.br_block_size)):
                    break
            block = _dedup_ints(block)
            if len(block) < 2:
                continue
            block_set = set(int(x) for x in block)

            # External centroid: where do these chips "want" to be, based on outside strong partners?
            ext_w: Dict[int, float] = {}
            for i in block:
                row = np.asarray(traffic_sym, dtype=np.float64)[int(i)]
                order = np.argsort(-row)
                taken = 0
                for j0 in order.tolist():
                    j = int(j0)
                    if j == int(i) or j in block_set:
                        continue
                    ext_w[j] = float(ext_w.get(j, 0.0)) + float(row[j])
                    taken += 1
                    if taken >= max(1, int(self.br_external_partner_topk)):
                        break
            ext_idxs = [int(x) for x in ext_w.keys()]
            if ext_idxs:
                ww = np.asarray([float(ext_w[j]) for j in ext_idxs], dtype=np.float64)
                # NOTE:
                # ext_idxs are global chip indices, while ww is a local weight vector aligned
                # with ext_idxs (not a global length-S vector). _weighted_centroid() now supports
                # this aligned form directly.
                centroid = self._weighted_centroid(pos, ext_idxs, ww)
            else:
                centroid = np.asarray(pos[np.asarray(block, dtype=int)], dtype=np.float64).mean(axis=0)

            # Prefer relocating into nearby empty sites; fallback to all sites if not enough empties.
            if bool(self.br_use_empty_only) and int(empty.size) >= len(block):
                site_pool = np.asarray(empty, dtype=int)
            else:
                site_pool = np.asarray(all_sites, dtype=int)

            anchors = _nearest_sites(
                np.asarray(sites_xy_mm, dtype=np.float64),
                centroid,
                site_pool,
                max(len(block), int(self.br_anchor_topk)),
            )
            anchors = _dedup_ints([int(x) for x in anchors])

            # Precompute "before" structural terms for this block.
            before_sites = [int(a[int(i)]) for i in block]
            before_pos = np.asarray(sites_xy_mm, dtype=np.float64)[np.asarray(before_sites, dtype=int)]
            compact_before = float(np.mean(np.sum((before_pos - before_pos.mean(axis=0, keepdims=True)) ** 2, axis=1)))
            cent_before = float(np.mean(np.sum((before_pos - centroid[None, :]) ** 2, axis=1)))
            tr_block = np.asarray(traffic_sym, dtype=np.float64)[np.ix_(block, block)]
            internal_before = _pairwise_comm_cost(before_pos, list(range(len(block))), tr_block)

            for anchor in anchors[: max(1, int(self.br_anchor_topk))]:
                # Candidate target pool around anchor. We allow a few extra sites and then
                # search a small permutation set to assign chips->sites (block_size is small).
                m = max(len(block), int(self.br_perm_sites_k))
                tgt0 = _nearest_sites(
                    np.asarray(sites_xy_mm, dtype=np.float64),
                    np.asarray(sites_xy_mm, dtype=np.float64)[int(anchor)],
                    site_pool,
                    m,
                )
                tgt0 = _dedup_ints([int(x) for x in tgt0])
                if len(tgt0) < len(block):
                    continue
                if set(tgt0[: len(block)]) == set(before_sites):
                    continue

                best_score: Optional[float] = None
                best_acts: List[Dict[str, Any]] = []
                best_cur = a.copy()
                perm_count = 0

                # Permute assignment of block chips to nearby sites.
                for perm in permutations(tgt0[:m], len(block)):
                    perm_count += 1
                    if perm_count > max(1, int(self.br_perm_limit)):
                        break
                    if set(int(x) for x in perm) == set(before_sites):
                        continue

                    cur = a.copy()
                    acts: List[Dict[str, Any]] = []
                    est_sum = 0.0
                    for i, sid in zip(block, perm):
                        act = {"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"}
                        est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                        est_sum += float(est.get("d_total", 0.0))
                        acts.append(dict(act))
                        cur = self._apply_act(cur, act)
                    if len(acts) < 2:
                        continue

                    after_sites = [int(cur[int(i)]) for i in block]
                    after_pos = np.asarray(sites_xy_mm, dtype=np.float64)[np.asarray(after_sites, dtype=int)]
                    compact_after = float(np.mean(np.sum((after_pos - after_pos.mean(axis=0, keepdims=True)) ** 2, axis=1)))
                    cent_after = float(np.mean(np.sum((after_pos - centroid[None, :]) ** 2, axis=1)))
                    internal_after = _pairwise_comm_cost(after_pos, list(range(len(block))), tr_block)

                    score = (
                        float(self.br_score_w_est) * float(est_sum)
                        + float(self.br_score_w_internal) * float(internal_after - internal_before)
                        + float(self.br_score_w_compact) * float(compact_after - compact_before)
                        + float(self.br_score_w_centroid) * float(cent_after - cent_before)
                    )
                    if best_score is None or float(score) < float(best_score):
                        best_score = float(score)
                        best_acts = [dict(x) for x in acts]
                        best_cur = cur.copy()

                if best_score is None or len(best_acts) < 2:
                    continue
                candidates.append((float(best_score), best_acts, best_cur))
                if len(candidates) >= max(4, int(self.br_candidate_blocks)):
                    break
            if len(candidates) >= max(4, int(self.br_candidate_blocks)):
                break

        candidates.sort(key=lambda x: float(x[0]))
        return candidates[: max(1, int(self.br_candidate_blocks))]

    def _run_block_relocate(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        candidates = self._propose_block_relocate_candidates(
            assign=assign0,
            sites_xy_mm=sites_xy_mm,
            traffic_sym=traffic_sym,
            chip_tdp_w=chip_tdp_w,
            site_to_region=site_to_region,
        )
        if not candidates:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        k_eval = max(1, min(int(self.br_eval_topk), len(candidates)))

        best_total = float(eval0.get("total_scalar", 0.0))
        best_assign = assign0.copy()
        best_eval = dict(eval0)
        best_acts: List[Dict[str, Any]] = []
        best_final = assign0.copy()

        for _k in range(int(k_eval)):
            _est, acts, final_assign = candidates[_k]
            eo = evaluate_assign(final_assign)
            info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
            v = float(eo.get("total_scalar", 1e30))
            if v < best_total:
                best_total = float(v)
                best_assign = final_assign.copy()
                best_eval = dict(eo)
                best_acts = [dict(a) for a in acts]
                best_final = final_assign.copy()

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(best_acts))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), best_acts, best_final, info

    def _run_chain(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
        n_steps: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        chain_len = max(2, int(self.chain_len), int(n_steps))
        n_prop = max(4, int(self.chain_propose))
        k_eval = max(1, min(int(self.chain_eval_topk), n_prop))

        cand_chains: List[Tuple[float, List[Dict[str, Any]], np.ndarray]] = []
        for _ in range(int(n_prop)):
            cur = assign0.copy()
            acts: List[Dict[str, Any]] = []
            est_sum = 0.0
            for _t in range(int(chain_len)):
                pool = self._propose_chain_primitives(cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region)
                if not pool:
                    break
                scored: List[Tuple[float, Dict[str, Any]]] = []
                for act in pool:
                    est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                    scored.append((float(est.get("d_total", 0.0)), dict(act)))
                scored.sort(key=lambda x: float(x[0]))
                topk = scored[: max(1, min(int(self.chain_step_topk), len(scored)))]
                samp_n = max(1, min(int(self.chain_step_sample_topk), len(topk)))
                pick = topk[_rng_randint(self.rng, 0, samp_n)] if samp_n > 1 else topk[0]
                d0, a0 = float(pick[0]), dict(pick[1])
                est_sum += d0
                acts.append(a0)
                cur = self._apply_act(cur, a0)
            if acts:
                cand_chains.append((float(est_sum), acts, cur))

        if not cand_chains:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        cand_chains.sort(key=lambda x: float(x[0]))

        best_total = float(eval0.get("total_scalar", 0.0))
        best_assign = assign0.copy()
        best_eval = dict(eval0)
        best_acts: List[Dict[str, Any]] = []
        best_final = assign0.copy()

        for _k in range(int(k_eval)):
            _est, acts, final_assign = cand_chains[_k]
            eo = evaluate_assign(final_assign)
            info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
            v = float(eo.get("total_scalar", 1e30))
            if v < best_total:
                best_total = float(v)
                best_assign = final_assign.copy()
                best_eval = dict(eo)
                best_acts = [dict(a) for a in acts]
                best_final = final_assign.copy()

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(best_acts))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), best_acts, best_final, info

    def _run_ruin_repair(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        S = int(assign0.shape[0])
        Ns = int(sites_xy_mm.shape[0])
        empty0 = self._empty_sites(assign0, Ns)

        # Target hard parts: traffic-hot + thermal-hot chips.
        try:
            hot_comm = self._hot_slots_by_traffic(traffic_sym, max(2, int(self.rr_hot_comm_k)))
        except Exception:
            hot_comm = []
        try:
            tdp = np.asarray(chip_tdp_w, dtype=np.float64)
            hot_therm = [int(x) for x in np.argsort(-tdp)[: max(2, min(S, int(self.rr_hot_therm_k)))].tolist()]
        except Exception:
            hot_therm = []
        hot_pool = _dedup_ints([int(x) for x in (hot_comm + hot_therm)])

        pos0 = np.asarray(sites_xy_mm, dtype=np.float64)[np.asarray(assign0, dtype=int)]

        ratios = [r for r in self.rr_ratios if r > 0.0]
        if not ratios:
            ratios = [0.2]

        candidates: List[Tuple[float, List[Dict[str, Any]], np.ndarray]] = []
        for r in ratios:
            n_can = max(1, int(self.rr_candidates_per_ratio))
            n_ruin = max(1, min(S, int(round(float(r) * float(S)))))
            for _c in range(int(n_can)):
                cur = assign0.copy()
                acts: List[Dict[str, Any]] = []
                est_sum = 0.0

                # Build a unique ruin set (avoid repeatedly "ruining" the same chip).
                ruin_set: List[int] = []
                tries_guard = 0
                while len(ruin_set) < int(n_ruin) and tries_guard < int(n_ruin) * 4:
                    tries_guard += 1
                    use_hot = bool(hot_pool) and (
                        float(getattr(self.rng, "random", lambda: np.random.rand())()) < float(self.rr_hot_prob)
                    )
                    if use_hot:
                        i = int(hot_pool[_rng_randint(self.rng, 0, len(hot_pool))])
                    else:
                        i = _rng_randint(self.rng, 0, S)
                    if i not in ruin_set:
                        ruin_set.append(int(i))

                for i in ruin_set:
                    if empty0.size > 0:
                        if str(self.rr_ruin_site_mode) == "far":
                            far = _farthest_sites(
                                np.asarray(sites_xy_mm, dtype=np.float64),
                                np.asarray(pos0[int(i)], dtype=np.float64),
                                np.asarray(empty0, dtype=int),
                                max(1, min(int(self.rr_far_site_k), int(empty0.size))),
                            )
                            if far:
                                pick_n = max(1, min(5, len(far)))
                                sid = int(far[_rng_randint(self.rng, 0, pick_n)])
                            else:
                                sid = int(empty0[_rng_randint(self.rng, 0, empty0.size)])
                        else:
                            sid = int(empty0[_rng_randint(self.rng, 0, empty0.size)])
                        act = {"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"}
                    else:
                        j = _rng_randint(self.rng, 0, S)
                        act = {"op": "swap", "i": int(i), "j": int(j), "type": "swap"}

                    try:
                        est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                        est_sum += float(est.get("d_total", 0.0))
                    except Exception:
                        pass
                    acts.append(dict(act))
                    cur = self._apply_act(cur, act)

                for _t in range(max(0, int(self.rr_repair_steps))):
                    pool = self._propose_chain_primitives(cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region)
                    if not pool:
                        break
                    scored: List[Tuple[float, Dict[str, Any]]] = []
                    for act in pool:
                        est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                        scored.append((float(est.get("d_total", 0.0)), dict(act)))
                    scored.sort(key=lambda x: float(x[0]))
                    topk = scored[: max(1, min(int(self.rr_repair_step_topk), len(scored)))]
                    if str(self.rr_repair_pick) == "best":
                        _d0, a0 = float(topk[0][0]), dict(topk[0][1])
                    else:
                        samp_n = max(1, min(int(self.rr_repair_step_sample_topk), len(topk)))
                        pick = topk[_rng_randint(self.rng, 0, samp_n)] if samp_n > 1 else topk[0]
                        _d0, a0 = float(pick[0]), dict(pick[1])
                    est_sum += float(_d0)
                    acts.append(dict(a0))
                    cur = self._apply_act(cur, a0)

                # Rank candidates by analytic delta sum (lower is better).
                est0 = float(est_sum) + 1e-6 * float(len(acts))
                candidates.append((float(est0), acts, cur))

        if not candidates:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        candidates.sort(key=lambda x: float(x[0]))
        k_eval = max(1, min(int(self.rr_eval_topk), len(candidates)))

        # Evaluate a few best candidates plus a tiny random tail for diversity.
        eval_random = max(0, min(int(self.rr_eval_random), max(0, k_eval - 1)))
        keep_top = max(1, k_eval - eval_random)
        eval_list = candidates[:keep_top]
        if eval_random > 0 and len(candidates) > keep_top:
            rest = candidates[keep_top:]
            for _ in range(int(eval_random)):
                eval_list.append(rest[_rng_randint(self.rng, 0, len(rest))])
        candidates = eval_list

        best_total = float(eval0.get("total_scalar", 0.0))
        best_assign = assign0.copy()
        best_eval = dict(eval0)
        best_acts: List[Dict[str, Any]] = []
        best_final = assign0.copy()

        for _k in range(int(k_eval)):
            _est, acts, final_assign = candidates[_k]
            eo = evaluate_assign(final_assign)
            info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
            v = float(eo.get("total_scalar", 1e30))
            if v < best_total:
                best_total = float(v)
                best_assign = final_assign.copy()
                best_eval = dict(eo)
                best_acts = [dict(a) for a in acts]
                best_final = final_assign.copy()

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(best_acts))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), best_acts, best_final, info

    def _hot_slots_by_traffic(self, traffic_sym: np.ndarray, k: int) -> List[int]:
        s = np.sum(np.asarray(traffic_sym, dtype=np.float64), axis=1)
        idx = np.argsort(-s)[: max(1, min(int(k), s.shape[0]))]
        return [int(x) for x in idx.tolist()]

    def _top_partners(self, traffic_sym: np.ndarray, i: int, k: int) -> List[int]:
        row = np.asarray(traffic_sym, dtype=np.float64)[int(i)].copy()
        row[int(i)] = -1.0
        idx = np.argsort(-row)[: max(1, min(int(k), row.shape[0] - 1))]
        return [int(x) for x in idx.tolist()]

    def _weighted_centroid(self, pos: np.ndarray, idxs: List[int], w: np.ndarray) -> np.ndarray:
        """
        Compute weighted centroid for positions selected by `idxs`.

        Supports two conventions for `w`:
          1) global weight vector: len(w) is larger than max(idxs), and weights are accessed by global idx
          2) local aligned weight vector: len(w) == len(idxs), where w[k] corresponds to idxs[k]

        The old implementation only supported (1), but block_relocate passes (2),
        which caused IndexError when a global idx exceeded len(w)-1.
        """
        if not idxs:
            return pos.mean(axis=0)

        idx_arr = np.asarray([int(j) for j in idxs], dtype=int)
        if idx_arr.size == 0:
            return pos.mean(axis=0)

        pts = pos[idx_arr]
        w_arr = np.asarray(w, dtype=np.float64).reshape(-1)

        # Case A: local aligned weights (most relevant for block_relocate)
        if int(w_arr.size) == int(idx_arr.size):
            ww = w_arr.copy()
        else:
            # Case B: global weight vector indexed by idxs
            valid = (idx_arr >= 0) & (idx_arr < int(w_arr.size))
            if not np.all(valid):
                idx_arr = idx_arr[valid]
                pts = pts[valid]
            if idx_arr.size == 0:
                return pos.mean(axis=0)
            ww = w_arr[idx_arr]

        ww = np.maximum(np.asarray(ww, dtype=np.float64), 1e-12)
        denom = float(np.sum(ww))
        if not np.isfinite(denom) or denom <= 0.0:
            return np.asarray(pts.mean(axis=0), dtype=np.float64)

        c = np.sum(pts * ww[:, None], axis=0) / denom
        return np.asarray(c, dtype=np.float64)

    def propose_actions(
        self,
        name: str,
        assign: np.ndarray,
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        a = np.asarray(assign, dtype=int)
        S = int(a.shape[0])
        Ns = int(sites_xy_mm.shape[0])
        pos = np.asarray(sites_xy_mm, dtype=np.float64)[a]
        empty = self._empty_sites(a, Ns)

        cands: List[Dict[str, Any]] = []

        if name == "comm":
            hot = self._hot_slots_by_traffic(traffic_sym, self.hot_slots_k)
            for i in hot:
                partners = self._top_partners(traffic_sym, i, self.partner_topk)
                wrow = np.asarray(traffic_sym, dtype=np.float64)[i]
                cent = self._weighted_centroid(pos, partners, wrow)
                if empty.size > 0:
                    near = _nearest_sites(np.asarray(sites_xy_mm, dtype=np.float64), cent, empty, self.empty_site_k)
                    for sid in near:
                        cands.append({"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"})
                for j in partners[: max(2, min(6, len(partners)))]:
                    if i != j:
                        cands.append({"op": "swap", "i": int(i), "j": int(j), "type": "swap"})

        elif name == "therm":
            tdp = np.asarray(chip_tdp_w, dtype=np.float64)
            hot_idx = np.argsort(-tdp)[: max(1, min(self.therm_hot_k, S))]
            cold_idx = np.argsort(tdp)[: max(1, min(self.therm_cold_k, S))]
            hot_slots = [int(x) for x in hot_idx.tolist()]
            cold_slots = [int(x) for x in cold_idx.tolist()]
            for i in hot_slots:
                di = np.sum((pos[np.asarray(cold_slots, dtype=int)] - pos[int(i)][None, :]) ** 2, axis=1)
                far_order = np.argsort(-di)[: max(1, min(6, di.shape[0]))]
                for t in far_order.tolist():
                    j = int(cold_slots[int(t)])
                    if i != j:
                        cands.append({"op": "swap", "i": int(i), "j": int(j), "type": "swap"})
                if empty.size > 0:
                    far_sites = _farthest_sites(np.asarray(sites_xy_mm, dtype=np.float64), pos[int(i)], empty, max(2, min(8, empty.size)))
                    for sid in far_sites:
                        cands.append({"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"})

        elif name == "escape":
            hot = self._hot_slots_by_traffic(traffic_sym, max(6, self.hot_slots_k // 2))
            tdp = np.asarray(chip_tdp_w, dtype=np.float64)
            hot_t = np.argsort(-tdp)[: max(1, min(6, S))]
            mix = list(dict.fromkeys([int(x) for x in hot] + [int(x) for x in hot_t.tolist()]))
            if empty.size > 0 and mix:
                for _ in range(min(6, len(mix))):
                    i = int(mix[_rng_randint(self.rng, 0, len(mix))])
                    sid = int(empty[_rng_randint(self.rng, 0, empty.size)])
                    cands.append({"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"})
            else:
                for _ in range(min(6, S)):
                    i = _rng_randint(self.rng, 0, S)
                    j = _rng_randint(self.rng, 0, S)
                    if i != j:
                        cands.append({"op": "swap", "i": int(i), "j": int(j), "type": "swap"})

        else:
            hot = self._hot_slots_by_traffic(traffic_sym, max(6, self.hot_slots_k // 2))
            for i in hot:
                partners = self._top_partners(traffic_sym, i, max(2, min(6, self.partner_topk)))
                for j in partners:
                    if i != j:
                        cands.append({"op": "swap", "i": int(i), "j": int(j), "type": "swap"})

        # de-dup + cap
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for act in cands:
            op = str(act.get("op"))
            if op == "swap":
                i = int(act.get("i", -1)); j = int(act.get("j", -1))
                a0, b0 = (i, j) if i <= j else (j, i)
                key = ("swap", a0, b0)
            elif op == "relocate":
                key = ("relocate", int(act.get("i", -1)), int(act.get("site_id", -1)))
            else:
                key = (op,)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(act)
            if len(uniq) >= max(10, self.cand_k):
                break
        return uniq

    # ----------------------------
    # Component A: Elite archive + Path Relinking
    # ----------------------------
    def _run_relink(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
        n_steps: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        tgt = self._pick_elite_target(assign0)
        if tgt is None:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        cur = assign0.copy()
        cur_eval = dict(eval0)
        best_total = float(cur_eval.get("total_scalar", 0.0))
        best_assign = cur.copy()
        best_eval = dict(cur_eval)
        executed: List[Dict[str, Any]] = []

        S = int(cur.shape[0])
        hot_comm = self._hot_slots_by_traffic(traffic_sym, max(8, self.hot_slots_k))
        tdp = np.asarray(chip_tdp_w, dtype=np.float64)
        hot_therm = [int(x) for x in np.argsort(-tdp)[: max(6, min(S, self.therm_hot_k))].tolist()]
        pri = list(dict.fromkeys([int(x) for x in hot_comm] + hot_therm))

        for _t in range(max(1, int(n_steps))):
            diff = np.nonzero(np.asarray(cur, dtype=int) != np.asarray(tgt.assign, dtype=int))[0].astype(int).tolist()
            if not diff:
                break
            cand_chips: List[int] = []
            for i in pri:
                if i in diff:
                    cand_chips.append(int(i))
                if len(cand_chips) >= max(4, int(self.rlk_pick_k)):
                    break
            if len(cand_chips) < max(4, int(self.rlk_pick_k)):
                self.rng.shuffle(diff)
                for i in diff:
                    if i not in cand_chips:
                        cand_chips.append(int(i))
                    if len(cand_chips) >= max(4, int(self.rlk_pick_k)):
                        break

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for i in cand_chips[: max(1, int(self.rlk_moves_per_step))]:
                sid = int(np.asarray(tgt.assign, dtype=int)[int(i)])
                act = {"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relink"}
                est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                act2 = dict(act); act2["_est"] = dict(est)
                scored.append((float(est.get("d_total", 0.0)), act2))
            scored.sort(key=lambda x: float(x[0]))
            if not scored:
                break

            k_eval = max(1, min(int(self.rlk_eval_topk), len(scored)))
            best_step_total = float(cur_eval.get("total_scalar", 0.0))
            best_step_assign = None
            best_step_eval = None
            best_step_act = None
            for _k in range(int(k_eval)):
                act = scored[_k][1]
                trial = _apply_relocate_perm(cur, int(act.get("i", 0)), int(act.get("site_id", 0)))
                eo = evaluate_assign(trial)
                info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
                v = float(eo.get("total_scalar", 1e30))
                if v < best_step_total:
                    best_step_total = v
                    best_step_assign = trial
                    best_step_eval = eo
                    best_step_act = act
            if best_step_assign is None:
                break

            if (best_step_total <= float(cur_eval.get("total_scalar", 0.0)) + 1e-12) or bool(self.rlk_allow_worsen):
                cur = np.asarray(best_step_assign, dtype=int).copy()
                cur_eval = dict(best_step_eval or cur_eval)
                executed.append(dict(best_step_act or {}))
                if best_step_total < best_total:
                    best_total = float(best_step_total)
                    best_assign = cur.copy()
                    best_eval = dict(cur_eval)
                try:
                    self.observe_elite(cur, cur_eval)
                except Exception:
                    pass
            else:
                break

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(executed))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), executed, cur.copy(), info

    # ----------------------------
    # Component B: ILS-style Shake (perturb + short repair)
    # ----------------------------
    def _run_shake(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        a0 = np.asarray(assign0, dtype=int).copy()
        # NOTE:
        # Old shake kick stage used fully random relocations (hot chips -> random sites).
        # In this task it almost always destroys structure (esp. under boundary-like penalties),
        # and the short repair cannot recover => gain==0 => success==0 across the grid.
        #
        # Fix:
        # Make kick stage structured BUT still stochastic:
        #   - Sample candidates from propose_actions("escape"/"comm"/"therm")
        #   - Score by estimate_action_delta
        #   - Randomly pick within top-K (K = shake_eval_topk) to keep exploration
        # Repair stage becomes multi-objective (comm + therm).

        best_total = float(eval0.get("total_scalar", 0.0))
        best_assign = a0.copy()
        best_eval = dict(eval0)
        best_acts: List[Dict[str, Any]] = []
        best_final = a0.copy()

        for _c in range(max(1, int(self.shake_candidates))):
            cur = a0.copy()
            acts: List[Dict[str, Any]] = []
            for _r in range(max(1, int(self.shake_kick_rounds))):
                # Build a structured candidate pool for the kick.
                kick_cand: List[Dict[str, Any]] = []
                try:
                    kick_cand.extend(self.propose_actions("escape", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
                except Exception:
                    pass
                try:
                    kick_cand.extend(self.propose_actions("comm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
                except Exception:
                    pass
                try:
                    kick_cand.extend(self.propose_actions("therm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
                except Exception:
                    pass

                # Fallback: if propose_actions returns nothing (should be rare), do a mild random swap.
                if not kick_cand:
                    i = int(_rng_randint(self.rng, 0, int(cur.shape[0])))
                    j = int(_rng_randint(self.rng, 0, int(cur.shape[0])))
                    if i != j:
                        act = {"op": "swap", "i": int(i), "j": int(j), "type": "shake"}
                        cur = self._apply_act(cur, act)
                        acts.append(dict(act))
                    continue

                # Score candidates by estimated delta (lower is better).
                scored_kick: List[Tuple[float, Dict[str, Any]]] = []
                for act in kick_cand:
                    est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                    act2 = dict(act)
                    act2["type"] = "shake"  # normalize type for logging/analysis
                    act2["_est"] = dict(est)
                    scored_kick.append((float(est.get("d_total", 0.0)), act2))
                scored_kick.sort(key=lambda x: float(x[0]))

                # Apply a few stochastic kicks: random pick inside top-K (K = shake_eval_topk).
                k_pool = max(1, min(int(self.shake_eval_topk), len(scored_kick)))
                for _ in range(max(1, int(self.shake_kick_k))):
                    pick = int(_rng_randint(self.rng, 0, int(k_pool)))
                    act = scored_kick[pick][1]
                    cur = self._apply_act(cur, act)
                    acts.append(dict(act))

            cur_eval = evaluate_assign(cur)
            info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
            for _rs in range(max(0, int(self.shake_repair_steps))):
                # Repair should be multi-objective (comm + therm) to recover from perturbations.
                cand: List[Dict[str, Any]] = []
                try:
                    cand.extend(self.propose_actions("comm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
                except Exception:
                    pass
                try:
                    cand.extend(self.propose_actions("therm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
                except Exception:
                    pass
                if not cand:
                    break
                scored: List[Tuple[float, Dict[str, Any]]] = []
                for act in cand:
                    est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                    act2 = dict(act); act2["_est"] = dict(est)
                    scored.append((float(est.get("d_total", 0.0)), act2))
                scored.sort(key=lambda x: float(x[0]))
                k_eval = max(1, min(int(self.shake_repair_eval_topk), len(scored)))
                best_step_total = float(cur_eval.get("total_scalar", 1e30))
                best_step = None
                best_step_eval = None
                best_act = None
                for _k in range(int(k_eval)):
                    act = scored[_k][1]
                    trial = self._apply_act(cur, act)
                    eo = evaluate_assign(trial)
                    info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
                    v = float(eo.get("total_scalar", 1e30))
                    if v < best_step_total:
                        best_step_total = v
                        best_step = trial
                        best_step_eval = eo
                        best_act = act
                if best_step is None:
                    break
                if (best_step_total <= float(cur_eval.get("total_scalar", 0.0)) + 1e-12) or bool(self.shake_allow_worsen):
                    cur = np.asarray(best_step, dtype=int).copy()
                    cur_eval = dict(best_step_eval or cur_eval)
                    if best_act is not None:
                        acts.append(dict(best_act))
                else:
                    break

            v = float(cur_eval.get("total_scalar", 1e30))
            if v < best_total:
                best_total = float(v)
                best_assign = cur.copy()
                best_eval = dict(cur_eval)
                best_acts = [dict(a) for a in acts]
                best_final = cur.copy()

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(best_acts))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), best_acts, best_final, info

    # ----------------------------
    # Component C: Tabu Search (short-run)
    # ----------------------------
    def _run_tabu_search(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}
        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        st.tries += 1
        info["tries"] = int(info.get("tries", 0)) + 1

        cur = np.asarray(assign0, dtype=int).copy()
        cur_eval = dict(eval0)
        best_total = float(cur_eval.get("total_scalar", 0.0))
        best_assign = cur.copy()
        best_eval = dict(cur_eval)
        executed: List[Dict[str, Any]] = []

        tabu: Dict[int, int] = {}  # chip -> expiry_step
        stepN = max(1, int(self.tabu_steps))
        tenure = max(1, int(self.tabu_tenure))

        for t in range(stepN):
            cand = []
            try:
                cand.extend(self.propose_actions("comm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
            except Exception:
                pass
            try:
                cand.extend(self.propose_actions("therm", cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region))
            except Exception:
                pass
            if not cand:
                break

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for act in cand[: max(1, int(self.tabu_cand_k))]:
                op = str(act.get("op"))
                chips: List[int] = []
                if op == "swap":
                    chips = [int(act.get("i", -1)), int(act.get("j", -1))]
                elif op == "relocate":
                    chips = [int(act.get("i", -1))]
                is_tabu = False
                for c in chips:
                    if c >= 0 and int(tabu.get(int(c), -1)) > int(t):
                        is_tabu = True
                        break
                est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                d = float(est.get("d_total", 0.0))
                if is_tabu and d >= -float(self.tabu_aspire_eps):
                    continue
                act2 = dict(act); act2["_est"] = dict(est)
                scored.append((d, act2))
            if not scored:
                break
            scored.sort(key=lambda x: float(x[0]))

            k_eval = max(1, min(int(self.tabu_eval_topk), len(scored)))
            best_step_total = float(cur_eval.get("total_scalar", 1e30))
            best_step = None
            best_step_eval = None
            best_act = None
            for _k in range(int(k_eval)):
                act = scored[_k][1]
                trial = self._apply_act(cur, act)
                eo = evaluate_assign(trial)
                info["eval_calls"] = int(info.get("eval_calls", 0)) + 1
                v = float(eo.get("total_scalar", 1e30))
                if v < best_step_total:
                    best_step_total = v
                    best_step = trial
                    best_step_eval = eo
                    best_act = act
            if best_step is None:
                break

            if (best_step_total <= float(cur_eval.get("total_scalar", 0.0)) + 1e-12) or bool(self.tabu_allow_worsen):
                cur = np.asarray(best_step, dtype=int).copy()
                cur_eval = dict(best_step_eval or cur_eval)
                if best_act is not None:
                    executed.append(dict(best_act))
                    op = str(best_act.get("op"))
                    touched: List[int] = []
                    if op == "swap":
                        touched = [int(best_act.get("i", -1)), int(best_act.get("j", -1))]
                    elif op == "relocate":
                        touched = [int(best_act.get("i", -1))]
                    for c in touched:
                        if c >= 0:
                            tabu[int(c)] = int(t) + int(tenure)
                if float(cur_eval.get("total_scalar", 1e30)) < best_total:
                    best_total = float(cur_eval.get("total_scalar", 1e30))
                    best_assign = cur.copy()
                    best_eval = dict(cur_eval)
            else:
                break

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = float(info.get("eval_calls", 0))
        info["used_actions"] = int(len(executed))
        self._update_stat_end(st, gain=gain, calls=calls, info=info)
        return best_assign, best_eval, float(best_total), executed, cur.copy(), info

    def run_macro(
        self,
        name: str,
        assign0: np.ndarray,
        eval0: Dict[str, Any],
        sites_xy_mm: np.ndarray,
        traffic_sym: np.ndarray,
        chip_tdp_w: np.ndarray,
        site_to_region: Optional[np.ndarray],
        evaluate_assign: Callable[[np.ndarray], Dict[str, Any]],
        n_steps: int = 3,
        max_eval_per_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        if str(name) in {"relink", "elite_relink", "path_relink"}:
            return self._run_relink(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
                n_steps=int(n_steps),
            )
        if str(name) in {"shake", "ils_shake", "kick_repair"}:
            return self._run_shake(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
            )
        if str(name) in {"tabu", "tabu_search", "tabu_ls"}:
            return self._run_tabu_search(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
            )
        if str(name) in {"chain", "macro_chain"}:
            return self._run_chain(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
                n_steps=int(n_steps),
            )
        if str(name) in {"ruin_repair", "ruin-and-recreate", "ruin"}:
            return self._run_ruin_repair(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
            )
        if str(name) in {"block_relocate", "block", "block-relocate"}:
            return self._run_block_relocate(
                name=str(name),
                assign0=assign0,
                eval0=eval0,
                sites_xy_mm=sites_xy_mm,
                traffic_sym=traffic_sym,
                chip_tdp_w=chip_tdp_w,
                site_to_region=site_to_region,
                evaluate_assign=evaluate_assign,
            )

        st = self._stat(name)
        info: Dict[str, Any] = {"name": str(name), "tries": 0, "success": 0, "fail": 0, "eval_calls": 0, "used_actions": 0}

        if st.cooldown > 0:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        cur = assign0.copy()
        cur_eval = dict(eval0)
        best_total = float(cur_eval.get("total_scalar", 0.0))
        best_assign = cur.copy()
        best_eval = dict(cur_eval)
        executed: List[Dict[str, Any]] = []

        for _t in range(max(1, int(n_steps))):
            st.tries += 1
            info["tries"] = int(info.get("tries", 0)) + 1

            cand_actions = self.propose_actions(name, cur, sites_xy_mm, traffic_sym, chip_tdp_w, site_to_region)
            if not cand_actions:
                break

            # rank by analytic delta
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for act in cand_actions:
                est = estimate_action_delta(cur, act, sites_xy_mm, traffic_sym, chip_tdp_w, self.obj)
                act2 = dict(act)
                act2["_est"] = est
                scored.append((float(est.get("d_total", 0.0)), act2))
            scored.sort(key=lambda x: float(x[0]))

            # verify top-k with real evaluator
            k_eval = max(1, min(int(self.eval_topk), len(scored)))
            if max_eval_per_step is not None:
                k_eval = max(1, min(k_eval, int(max_eval_per_step)))

            best_step = None
            best_step_eval = None
            best_step_total = float(cur_eval.get("total_scalar", 0.0))
            best_step_act = None

            for _k in range(k_eval):
                act = scored[_k][1]
                op = str(act.get("op"))
                if op == "swap":
                    trial = _apply_swap(cur, int(act.get("i", 0)), int(act.get("j", 0)))
                elif op == "relocate":
                    trial = _apply_relocate_perm(cur, int(act.get("i", 0)), int(act.get("site_id", 0)))
                else:
                    continue

                eo = evaluate_assign(trial)
                info["eval_calls"] = int(info.get("eval_calls", 0)) + 1

                v = float(eo.get("total_scalar", 1e30))
                if v < best_step_total:
                    best_step_total = v
                    best_step = trial
                    best_step_eval = eo
                    best_step_act = act

            if best_step is None or best_step_eval is None:
                break

            # monotone: only apply improving step
            if self.monotone and not (best_step_total < float(cur_eval.get("total_scalar", 0.0)) - 1e-12):
                st.fail += 1
                info["fail"] = int(info.get("fail", 0)) + 1
                if self.adapt_enabled:
                    st.cooldown = max(st.cooldown, int(self.fail_cooldown))
                    st.weight = max(self.weight_floor, float(st.weight) * (1.0 - self.fail_penalty))
                break

            cur = best_step
            cur_eval = dict(best_step_eval)
            executed.append({k: v for k, v in best_step_act.items() if not str(k).startswith("_")})
            info["used_actions"] = int(info.get("used_actions", 0)) + 1

            if best_step_total < best_total:
                best_total = float(best_step_total)
                best_assign = cur.copy()
                best_eval = dict(cur_eval)

        gain = max(0.0, float(eval0.get("total_scalar", 0.0)) - float(best_total))
        calls = max(1.0, float(info.get("eval_calls", 0)))

        if gain > 0.0:
            st.success += 1
            info["success"] = int(info.get("success", 0)) + 1
            if self.adapt_enabled:
                st.cooldown = max(st.cooldown, int(self.success_cooldown))
                st.weight = min(self.weight_cap, float(st.weight) * (1.0 + self.success_boost))
        else:
            st.fail += 1
            info["fail"] = int(info.get("fail", 0)) + 1
            if self.adapt_enabled:
                st.cooldown = max(st.cooldown, int(self.fail_cooldown))
                st.weight = max(self.weight_floor, float(st.weight) * (1.0 - self.fail_penalty))

        a = float(self.alpha)
        st.ewma_gain = (1.0 - a) * float(st.ewma_gain) + a * float(gain)
        st.ewma_calls = (1.0 - a) * float(st.ewma_calls) + a * float(calls)
        st.ewma_gain_per_call = float(st.ewma_gain) / float(max(1e-9, st.ewma_calls))

        return best_assign, best_eval, float(best_total), executed, cur.copy(), info
