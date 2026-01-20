from __future__ import annotations

from typing import Any, Dict


def resolve_lambda_hw(cfg: Any, stable_hw_state: Dict[str, Any] | None) -> float:
    """
    v5.4 Single source of truth:
      - if StableHW enabled: ALWAYS use stable_hw_state["lambda_hw_effective"]
        (already includes Acc-First Hard Gating g_hw)
      - else: use cfg.hw.lambda_hw (legacy)
    NoDrift: NEVER copy cfg.loss.lambda_hw into cfg.hw.lambda_hw here.
    """
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    stable_enabled = bool(getattr(stable_hw_cfg, "enabled", False)) if stable_hw_cfg is not None else False

    if stable_enabled:
        if not stable_hw_state:
            # StableHW enabled but state missing => safest is to hard-disable HW term.
            return 0.0
        return float(stable_hw_state.get("lambda_hw_effective", 0.0))

    # legacy
    hw_cfg = getattr(cfg, "hw", None)
    if hw_cfg is None:
        return 0.0
    return float(getattr(hw_cfg, "lambda_hw", 0.0))
