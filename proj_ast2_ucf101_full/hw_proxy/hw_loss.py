from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def compute_hw_loss(
    cfg: Any,
    hw_proxy: Any,
    model_info: Dict[str, Any],
    stable_hw_cfg: Optional[Dict[str, Any]] = None,
    stable_hw_state: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns: (hw_loss_scalar, metrics_dict)
    metrics_dict must at least include: hw_latency_ms, hw_mem_mb (if available), hw_energy_mj (optional)
    This is a thin wrapper: actual proxy math stays in hw_proxy implementation.
    """
    # ---- extract keep factors from model_info (robust defaults) ----
    token_keep = float(model_info.get("token_keep", model_info.get("token_keep_ratio", 1.0)) or 1.0)
    head_keep = float(model_info.get("head_keep", 1.0) or 1.0)
    ch_keep = float(model_info.get("ch_keep", 1.0) or 1.0)
    block_keep = float(model_info.get("block_keep", 1.0) or 1.0)

    # ---- call proxy (must be implemented in your hw_proxy) ----
    # Expect: dict with latency_ms, mem_mb, energy_mj (some may be missing)
    pred = hw_proxy.predict_from_model_info(
        model_info=model_info,
        token_keep=token_keep,
        head_keep=head_keep,
        ch_keep=ch_keep,
        block_keep=block_keep,
    )

    lat = float(pred.get("latency_ms", pred.get("lat_ms", 0.0)) or 0.0)
    mem = float(pred.get("mem_mb", pred.get("peak_mem_mb", 0.0)) or 0.0)
    eng = float(pred.get("energy_mj", pred.get("energy", 0.0)) or 0.0)

    metrics = {"hw_latency_ms": lat, "hw_mem_mb": mem, "hw_energy_mj": eng}

    # ---- stable_hw normalization (hinge_log_ratio default) ----
    # If stable_hw_cfg absent: use raw latency as loss.
    loss = lat
    if stable_hw_cfg is not None:
        norm = stable_hw_cfg.get("normalize", {}) if isinstance(stable_hw_cfg, dict) else {}
        mode = str(norm.get("mode", "hinge_log_ratio"))
        # baseline targets
        targ = stable_hw_cfg.get("targets", {}) if isinstance(stable_hw_cfg, dict) else {}
        lat0 = float(targ.get("latency_ms_ref", max(1e-6, lat)) or max(1e-6, lat))
        if mode == "hinge_log_ratio":
            import math

            ratio = max(1e-6, lat / max(1e-6, lat0))
            loss = max(0.0, math.log(ratio))
        # else: fallback raw
    return float(loss), metrics
