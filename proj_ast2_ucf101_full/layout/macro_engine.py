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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from layout.delta_eval import ObjectiveParams, estimate_action_delta


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


@dataclass
class OpStat:
    tries: int = 0
    success: int = 0
    fail: int = 0
    ewma_gain_per_call: float = 0.0
    ewma_gain: float = 0.0
    ewma_calls: float = 0.0
    cooldown: int = 0
    weight: float = 1.0


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
                "cooldown": int(st.cooldown),
                "weight": float(st.weight),
            }
        return out

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
                for _t in range(int(n_ruin)):
                    i = _rng_randint(self.rng, 0, S)
                    if empty0.size > 0:
                        sid = int(empty0[_rng_randint(self.rng, 0, empty0.size)])
                        act = {"op": "relocate", "i": int(i), "site_id": int(sid), "type": "relocate"}
                    else:
                        j = _rng_randint(self.rng, 0, S)
                        act = {"op": "swap", "i": int(i), "j": int(j), "type": "swap"}
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
                    samp_n = max(1, min(int(self.rr_repair_step_sample_topk), len(topk)))
                    pick = topk[_rng_randint(self.rng, 0, samp_n)] if samp_n > 1 else topk[0]
                    _d0, a0 = float(pick[0]), dict(pick[1])
                    acts.append(dict(a0))
                    cur = self._apply_act(cur, a0)

                est0 = float(r) + 1e-6 * float(len(acts))
                candidates.append((est0, acts, cur))

        if not candidates:
            return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy(), info

        k_eval = max(1, min(int(self.rr_eval_topk), len(candidates)))
        try:
            idxs = list(range(len(candidates)))
            for i in range(len(idxs) - 1, 0, -1):
                j = _rng_randint(self.rng, 0, i + 1)
                idxs[i], idxs[j] = idxs[j], idxs[i]
            candidates = [candidates[i] for i in idxs]
        except Exception:
            pass

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
        if not idxs:
            return pos.mean(axis=0)
        ww = np.asarray([float(w[int(j)]) for j in idxs], dtype=np.float64)
        ww = np.maximum(ww, 1e-12)
        pts = pos[np.asarray(idxs, dtype=int)]
        c = np.sum(pts * ww[:, None], axis=0) / float(np.sum(ww))
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
