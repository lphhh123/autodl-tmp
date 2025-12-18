"""Distributed training helpers (stubs for single-node experiments)."""
from __future__ import annotations

import torch


def get_device(device_str: str | None = None) -> torch.device:
    if device_str and device_str != "cuda if available":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
