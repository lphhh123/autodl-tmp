from __future__ import annotations

from typing import Any, Dict, Optional


def _resolve_seed_id(cfg: Any) -> int:
    for path in ("seed", "train", "training"):
        try:
            if path == "seed":
                seed_val = getattr(cfg, "seed", None)
            else:
                seed_val = getattr(getattr(cfg, path, None), "seed", None)
            if seed_val is not None:
                return int(seed_val or 0)
        except Exception:
            continue
    return 0


def make_gating_payload_v54(
    *,
    cfg: Any,
    stable_state: Dict[str, Any],
    epoch: int,
    step: int,
    loss_scalar: float,
    gate_ok: bool,
    gate_reason: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    overrides = dict(overrides or {})
    outer_iter = int(overrides.pop("outer_iter", epoch))
    inner_step = int(overrides.pop("inner_step", step))
    seed_id = int(overrides.pop("seed_id", _resolve_seed_id(cfg)))
    candidate_id = overrides.pop("candidate_id", f"single_device_ep{int(epoch):04d}_st{int(step):06d}")
    acc_now = float(stable_state.get("acc_now", 0.0) or 0.0)
    acc_used_value = stable_state.get("acc_used_value", None)
    acc_used = float(acc_used_value if acc_used_value is not None else acc_now)
    hw_metric_normed = dict(stable_state.get("hw_metric_normed", {}) or {})

    payload = {
        # required ids
        "outer_iter": int(outer_iter),
        "inner_step": int(inner_step),
        "seed_id": int(seed_id),
        "candidate_id": candidate_id,

        # decision
        "gate": "allow_hw" if bool(gate_ok) else "reject_hw",
        "reason_code": str(gate_reason),
        "guard_mode": str(stable_state.get("guard_mode", "UNKNOWN")),

        # accuracy side
        "acc_ref": float(stable_state.get("acc_ref", 0.0) or 0.0),
        "acc_used": float(
            stable_state.get("acc_used_value", None)
            if stable_state.get("acc_used_value", None) is not None
            else acc_now
        ),
        "acc_ref_source": str(stable_state.get("acc_ref_source", "unknown")),
        "acc_used_source": str(stable_state.get("acc_used_source", "unknown")),
        "acc_drop": float(stable_state.get("acc_drop", 0.0) or 0.0),
        "acc_drop_max": float(stable_state.get("acc_drop_max", 0.0) or 0.0),

        # hw side
        "lambda_hw_requested": float(stable_state.get("lambda_hw_requested", 0.0) or 0.0),
        "lambda_hw_effective": float(stable_state.get("lambda_hw_effective", 0.0) or 0.0),
        "hw_ref_source": str(stable_state.get("hw_ref_source", "unknown")),
        "hw_metric_ref": dict(stable_state.get("hw_metric_ref", {}) or {}),
        "hw_metric_raw": dict(stable_state.get("hw_metric_raw", {}) or {}),
        "hw_metric_normed": hw_metric_normed,
        "hw_metric_used": dict(stable_state.get("hw_metric_used", hw_metric_normed) or {}),
        "hw_metric_used_sanitized": dict(
            stable_state.get("hw_metric_used_sanitized", hw_metric_normed) or {}
        ),
        "hw_metric_key_order": list(stable_state.get("hw_metric_key_order", []) or []),

        "hw_loss_raw": float(stable_state.get("hw_loss_raw", 0.0) or 0.0),
        "hw_loss_used": float(
            stable_state.get("hw_loss_used", None)
            if stable_state.get("hw_loss_used", None) is not None
            else stable_state.get("hw_loss_raw", 0.0) or 0.0
        ),

        "hw_scale_penalty": float(stable_state.get("hw_scale_penalty", 0.0) or 0.0),
        "hw_scale_penalty_weight": float(stable_state.get("hw_scale_penalty_weight", 0.0) or 0.0),
        "hw_scale_penalty_applied": bool(stable_state.get("hw_scale_penalty_applied", False)),

        # loss decomposition (required by contract)
        "total_loss_hw_part": float(stable_state.get("total_loss_hw_part", 0.0) or 0.0),
        "total_loss_acc_part": float(stable_state.get("total_loss_acc_part", float(loss_scalar)) or float(loss_scalar)),
        "acc_weighted_loss": float(stable_state.get("acc_weighted_loss", float(loss_scalar)) or float(loss_scalar)),
        "hw_weighted_loss": float(stable_state.get("hw_weighted_loss", 0.0) or 0.0),
        "total_loss": float(stable_state.get("total_loss", float(loss_scalar)) or float(loss_scalar)),

        "hw_scale_schema_version": str(stable_state.get("hw_scale_schema_version", "v1")),

        # mapping/layout bookkeeping
        "mapping_id": int(stable_state.get("mapping_id", 0) or 0),
        "layout_id": int(stable_state.get("layout_id", 0) or 0),
        "mapping_signature": str(stable_state.get("mapping_signature", "")),
        "layout_signature": str(stable_state.get("layout_signature", "")),
        "is_valid_solution": True,

        "notes": "single_device",
    }

    payload["acc_used"] = acc_used
    payload.update(overrides)
    return payload
