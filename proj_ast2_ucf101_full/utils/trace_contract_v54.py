"""Shared contract constants for v5.4 trace events (SPEC_E)."""
from __future__ import annotations

REQUIRED_GATING_KEYS = {"gate", "acc_drop", "acc_drop_max"}
REQUIRED_PROXY_SANITIZE_KEYS = {"metric", "raw_value", "used_value", "penalty_added"}
