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
                "probe_min_samples": 2,
                "trial_max_per_step": 1,
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


def test_sponsored_trial_rules():
    c = _ctrl()
    ctx_mid = c.build_context(stagn=20, repeat_ratio=0.8, blocked_ratio=0.4, used_calls=50, budget_total=100)
    c.observe_probe("macro", "comm", ctx_mid["ctx_key"], 0.5, 0.1, 1, True, False)
    ok, _, _ = c.maybe_sponsor_trial("macro", "comm", ctx_mid, step=1)
    assert not ok

    ctx_late = c.build_context(stagn=40, repeat_ratio=0.8, blocked_ratio=0.4, used_calls=90, budget_total=100)
    c.observe_heuristic(gain=1.0, calls=1000)
    c.observe_probe("macro", "comm", ctx_late["ctx_key"], 0.0, 0.0, 5, True, False)
    ok, _, _ = c.maybe_sponsor_trial("macro", "comm", ctx_late, step=2)
    assert not ok

    c.observe_probe("macro", "comm", ctx_late["ctx_key"], 2.0, 0.1, 1, True, False)
    c.observe_probe("macro", "comm", ctx_late["ctx_key"], 2.0, 0.1, 1, True, False)
    ok, _, _ = c.maybe_sponsor_trial("macro", "comm", ctx_late, step=3)
    assert ok


def test_release_context_local_and_adjust_min_gain():
    c = _ctrl()
    ctx_late = c.build_context(stagn=40, repeat_ratio=0.8, blocked_ratio=0.4, used_calls=90, budget_total=100)
    ctx_other = c.build_context(stagn=5, repeat_ratio=0.2, blocked_ratio=0.0, used_calls=90, budget_total=100)
    for _ in range(3):
        c.observe_probe("macro", "comm", ctx_late["ctx_key"], 2.0, 0.3, 1, True, True)
    c.register_win("macro", used_calls=100, budget_total=1000, best_total_seen=10.0, ctx_key=ctx_late["ctx_key"], family="comm", sponsored=True)
    c.on_progress(used_calls=500, budget_total=1000, best_total_seen=0.0)

    assert c.release_active("macro", family="comm", ctx=ctx_late)
    assert not c.release_active("macro", family="escape", ctx=ctx_late)
    assert not c.release_active("macro", family="comm", ctx=ctx_other)

    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="comm", sponsored=True) == 0.0
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_late, family="comm", sponsored=False) == 0.0
    assert c.adjust_min_gain_current("macro", 0.001, ctx=ctx_other, family="comm", sponsored=False) >= 0.0
