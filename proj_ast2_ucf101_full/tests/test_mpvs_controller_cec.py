from layout.mpvs_controller import MPVSController


def _ctrl():
    return MPVSController(
        cfg={
            "ewma_alpha": 0.2,
            "share": {"slack_scale": 0.5, "min_samples": 1},
            "budget_stage": {"early_frac": 0.2, "late_frac": 0.7},
            "stagn_norm": 20,
            "cec": {
                "enabled": True,
                "family_blend_tau": 8,
                "family_min_samples": 6,
                "local_min_samples": 2,
                "seed_trials_per_family": 1,
                "family_cooldown_steps": 0,
                "trial_max_per_step": 1,
                "candidate_release": True,
                "family_stage_prior": True,
                "context": {
                    "stagn_ratio_edges": [0.75, 1.5],
                    "repeat_ratio_edges": [0.55, 0.75],
                    "blocked_ratio_edges": [0.15, 0.35],
                },
            },
            "macro": {"enabled": True, "quota_max": 5},
        }
    )


def test_build_context_buckets():
    c = _ctrl()
    ctx_e = c.build_context(stagn=5, repeat_ratio=0.2, blocked_ratio=0.05, used_calls=10, budget_total=100)
    ctx_m = c.build_context(stagn=20, repeat_ratio=0.6, blocked_ratio=0.2, used_calls=50, budget_total=100)
    ctx_l = c.build_context(stagn=40, repeat_ratio=0.8, blocked_ratio=0.4, used_calls=90, budget_total=100)
    assert ctx_e["stage"] == "early"
    assert ctx_m["stage"] == "mid"
    assert ctx_l["stage"] == "late"
    assert ctx_e["stagn_bucket"] < ctx_l["stagn_bucket"]
    assert ctx_e["health_bucket"] < ctx_l["health_bucket"]


def test_edge_score_dimless_supports_sponsor_when_positive():
    c = _ctrl()
    ctx_late = c.build_context(stagn=40, repeat_ratio=0.7, blocked_ratio=0.4, used_calls=95, budget_total=100)
    c.observe_heuristic(gain=10.0, calls=10000)
    for _ in range(7):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.01, margin_cur=-0.01, calls=1, pass_heur=True, pass_cur=False)
    ok, reason, meta = c.maybe_sponsor_trial("macro", "comm", ctx_late, step=1)
    assert ok
    assert reason in {"seed_sponsor", "evidence_sponsor"}
    assert float(meta["edge_local"]) > 0.0
    assert float(meta["trial_score"]) > 0.0


def test_seed_sponsor_uses_family_global_when_local_is_sparse():
    c = _ctrl()
    ctx_mid = c.build_context(stagn=20, repeat_ratio=0.7, blocked_ratio=0.4, used_calls=50, budget_total=100)
    ctx_late = c.build_context(stagn=40, repeat_ratio=0.7, blocked_ratio=0.4, used_calls=95, budget_total=100)
    c.observe_heuristic(gain=5.0, calls=5000)
    for _ in range(6):
        c.observe_probe("macro", "therm", ctx_mid["ctx_key"], margin_heur=0.02, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=False)
    c.observe_probe("macro", "therm", ctx_late["ctx_key"], margin_heur=0.02, margin_cur=-0.01, calls=1, pass_heur=True, pass_cur=False)
    ok, reason, meta = c.maybe_sponsor_trial("macro", "therm", ctx_late, step=2)
    assert ok
    assert reason == "seed_sponsor"
    assert int(meta["n_local"]) < c.cec_local_min_samples
    assert float(meta["edge_family"]) > 0.0


def test_evidence_sponsor_and_local_weight_growth():
    c = _ctrl()
    ctx_late = c.build_context(stagn=35, repeat_ratio=0.6, blocked_ratio=0.3, used_calls=92, budget_total=100)
    c.observe_heuristic(gain=5.0, calls=5000)
    for _ in range(8):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.03, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=False)
    score1 = c._trial_score("macro", ctx_late["ctx_key"], "comm")
    c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.03, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=False)
    score2 = c._trial_score("macro", ctx_late["ctx_key"], "comm")
    ok, reason, meta = c.maybe_sponsor_trial("macro", "comm", ctx_late, step=3)
    assert ok
    assert reason == "evidence_sponsor"
    assert float(score2["lambda_local"]) >= float(score1["lambda_local"])
    assert float(meta["trial_score"]) > 0.0


def test_release_stays_context_local():
    c = _ctrl()
    ctx_a = {"stage": "late", "ctx_key": "late|stg2|hlth2"}
    ctx_b = {"stage": "late", "ctx_key": "late|stg1|hlth1"}
    c.observe_heuristic(gain=1.0, calls=1000)
    for _ in range(8):
        c.observe_probe("macro", "comm", ctx_a["ctx_key"], margin_heur=0.03, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)
    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_a["ctx_key"], family="comm", sponsored=True)
    c.on_progress(used_calls=500, budget_total=1000, best_total_seen=0.0)
    assert c.release_active("macro", family="comm", ctx=ctx_a)
    assert not c.release_active("macro", family="comm", ctx=ctx_b)


def test_adjust_min_gain_current_rules():
    c = _ctrl()
    ctx_late = {"stage": "late", "ctx_key": "late|stg2|hlth2"}
    c.observe_heuristic(gain=1.0, calls=1000)
    for _ in range(8):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.03, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)
    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_late["ctx_key"], family="comm", sponsored=True)
    c.on_progress(used_calls=500, budget_total=1000, best_total_seen=0.0)

    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="comm", sponsored=True) == 0.0
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="comm", sponsored=False) == 0.0
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="other", sponsored=False) >= 0.0



def test_candidate_soft_release_enables_min_gain_before_horizon():
    c = _ctrl()
    ctx_late = {"stage": "late", "ctx_key": "late|stg2|hlth2"}
    c.observe_heuristic(gain=1.0, calls=1000)
    for _ in range(6):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.02, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)

    # Sponsored win activates candidate immediately.
    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_late["ctx_key"], family="comm", sponsored=True)
    assert c.candidate_active("macro", family="comm", ctx=ctx_late)
    # Candidate should relax strict current buffer to 0.0 (still requires non-negative gain later).
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="comm", sponsored=False) == 0.0
    # Candidate is ctx+family local
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="other", sponsored=False) >= 0.0


def test_candidate_expires_if_no_long_roi_at_horizon():
    c = _ctrl()
    ctx_late = {"stage": "late", "ctx_key": "late|stg2|hlth2"}
    c.observe_heuristic(gain=1.0, calls=1000)
    for _ in range(6):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.02, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)

    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_late["ctx_key"], family="comm", sponsored=True)
    assert c.candidate_active("macro", family="comm", ctx=ctx_late)

    # Horizon expires but best_total_seen doesn't improve => gain_long==0 => release shouldn't happen.
    c.on_progress(used_calls=500, budget_total=1000, best_total_seen=10.0)
    assert not c.candidate_active("macro", family="comm", ctx=ctx_late)
    assert not c.release_active("macro", family="comm", ctx=ctx_late)


def test_stage_family_prior_uses_mid_late_more_than_global_in_late():
    c = _ctrl()
    c.observe_heuristic(gain=5.0, calls=5000)

    ctx_early = c.build_context(stagn=5, repeat_ratio=0.2, blocked_ratio=0.05, used_calls=10, budget_total=100)
    ctx_mid = c.build_context(stagn=20, repeat_ratio=0.7, blocked_ratio=0.4, used_calls=50, budget_total=100)
    ctx_late = c.build_context(stagn=40, repeat_ratio=0.7, blocked_ratio=0.4, used_calls=95, budget_total=100)

    # Early negatives dilute global family, but should not dominate late prior.
    for _ in range(20):
        c.observe_probe("macro", "comm", ctx_early["ctx_key"], margin_heur=-0.02, margin_cur=-0.02, calls=1, pass_heur=False, pass_cur=False)
    for _ in range(6):
        c.observe_probe("macro", "comm", ctx_mid["ctx_key"], margin_heur=0.02, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=False)

    ts = c._trial_score("macro", ctx_late["ctx_key"], "comm")
    assert float(ts.get("edge_family", 0.0)) > float(ts.get("edge_family_global", -1.0))
    assert str(ts.get("prior_src", "")) in {"midlate", "midlate+global"}


def test_candidate_any_and_get_active_families_enable_source_level_use():
    c = _ctrl()
    ctx_late = {"stage": "late", "ctx_key": "late|stg2|hlth2"}
    c.observe_heuristic(gain=1.0, calls=1000)

    # Build a little probe history.
    for _ in range(6):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], margin_heur=0.02, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)
        c.observe_probe("macro", "therm", ctx_late["ctx_key"], margin_heur=0.01, margin_cur=0.0, calls=1, pass_heur=True, pass_cur=True)

    # Sponsored win -> candidate(comm)
    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_late["ctx_key"], family="comm", sponsored=True)

    # Simulate a released family for ordering check.
    a_th = c._get_ctx_agg("macro", ctx_late["ctx_key"], "therm")
    a_th.released = 1
    a_th.last_release_step = 200

    fams = c.get_active_families(ctx_key=ctx_late["ctx_key"], stage="late")
    assert fams[0] == "therm"
    assert "comm" in fams

    # Source-level candidate_any should be true even before choosing a concrete macro family.
    assert c.candidate_any("macro", ctx=ctx_late)

    # get_active_families/candidate_any should record hits for auditability.
    snap = c.snapshot()
    cec_ctx = snap.get("cec_ctx", {})
    cand_hits = 0
    rel_hits = 0
    for v in cec_ctx.values():
        cand_hits += int((v or {}).get("candidate_hits", 0))
        rel_hits += int((v or {}).get("release_hits", 0))
    assert cand_hits >= 1
    assert rel_hits >= 1
