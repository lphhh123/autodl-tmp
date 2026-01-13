from __future__ import annotations

from typing import Any, Optional

from .config_utils import get_nested, set_nested


def resolve_lambda_hw(cfg: Any, stable_hw_state: Optional[dict] = None) -> float:
    """
    Priority:
      1) stable_hw_state["lambda_hw"] if present
      2) cfg.hw.lambda_hw
      3) cfg.loss.lambda_hw (legacy compat)
      4) 0.0
    """
    if stable_hw_state is not None and isinstance(stable_hw_state, dict):
        v = stable_hw_state.get("lambda_hw", None)
        if v is not None:
            return float(v)

    v = get_nested(cfg, "hw.lambda_hw", None)
    if v is not None:
        return float(v)

    v = get_nested(cfg, "loss.lambda_hw", None)
    if v is not None:
        # normalize legacy -> new
        set_nested(cfg, "hw.lambda_hw", float(v))
        return float(v)

    set_nested(cfg, "hw.lambda_hw", 0.0)
    return 0.0
