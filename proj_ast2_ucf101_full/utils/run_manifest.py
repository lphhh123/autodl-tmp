import json
from pathlib import Path


def write_run_manifest(
    out_dir: str,
    cfg_path: str,
    cfg_hash: str,
    seed: int,
    stable_hw_state: dict,
    extra: dict | None = None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = {
        "cfg_path": str(cfg_path),
        "cfg_hash": str(cfg_hash),
        "seed": int(seed),
        "acc_ref": stable_hw_state.get("acc_ref"),
        "acc_ref_source": stable_hw_state.get("acc_ref_source"),
        "baseline_stats_path": (
            stable_hw_state.get("baseline_stats_path")
            or stable_hw_state.get("baseline_path")
            or stable_hw_state.get("baseline_stats")
        ),
        "epsilon_drop": stable_hw_state.get("epsilon_drop"),
        "schedule_phase": stable_hw_state.get("schedule_phase"),
        "mapping_signature": stable_hw_state.get("discrete_cache", {}).get("mapping_signature"),
        "layout_signature": stable_hw_state.get("discrete_cache", {}).get("layout_signature"),
        "stable_hw": {
            "guard_mode": stable_hw_state.get("guard_mode"),
            "lambda_hw_base": stable_hw_state.get("lambda_hw_base"),
            "lambda_hw_effective": stable_hw_state.get("lambda_hw_effective"),
        },
    }
    if extra:
        m.update(extra)
    (out / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
