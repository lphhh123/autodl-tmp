import numpy as np


def _make_symmetric(m: np.ndarray) -> np.ndarray:
    return m + m.T


def test_delta_swap_matches_full_evaluator():
    rng = np.random.default_rng(0)
    S = 16
    Ns = 40
    sites = rng.normal(size=(Ns, 2)).astype(np.float64)
    assign = rng.choice(np.arange(Ns), size=S, replace=False).astype(int)
    traffic = _make_symmetric(rng.random((S, S)).astype(np.float64))
    tdp = rng.random(S).astype(np.float64) * 2.0
    sigma = 1.7

    from layout.evaluator import LayoutEvaluator, LayoutState
    from layout.delta_eval import ObjectiveParams, delta_terms_swap, estimate_delta_norm_and_total

    ev = LayoutEvaluator(
        sigma_mm=float(sigma),
        baseline={"L_comm_baseline": 1.0, "L_therm_baseline": 1.0},
        scalar_w={"w_comm": 0.7, "w_therm": 0.3, "w_penalty": 0.0},
    )
    st = LayoutState(
        S=S,
        Ns=Ns,
        wafer_radius_mm=1e9,
        sites_xy_mm=sites,
        assign=assign.copy(),
        chip_tdp_w=tdp,
        traffic_bytes=traffic,
        meta={},
    )
    base = ev.evaluate(st)
    obj = ObjectiveParams(
        sigma_mm=sigma,
        L_comm_baseline=base["L_comm"],
        L_therm_baseline=base["L_therm"],
        w_comm=0.7,
        w_therm=0.3,
    )

    i, j = 3, 9
    a2 = assign.copy()
    a2[i], a2[j] = a2[j], a2[i]
    st.assign = a2
    after = ev.evaluate(st)

    dLc, dLt = delta_terms_swap(assign, i, j, sites, traffic, tdp, sigma)
    _, _, dt = estimate_delta_norm_and_total(dLc, dLt, obj)
    full_dt = float(after["total_scalar"]) - float(base["total_scalar"])

    assert np.isfinite(dt)
    assert abs(dt - full_dt) < 1e-6

