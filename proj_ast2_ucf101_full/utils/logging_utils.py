"""Logging utilities for training and evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch


def setup_logger(output_dir: str | Path | None = None) -> logging.Logger:
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ast2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def tensorboard_stub() -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception:
        return None
    return SummaryWriter


def log_stats(logger: logging.Logger, stats: Dict[str, float]) -> None:
    msg = ", ".join([f"{k}={v}" for k, v in stats.items()])
    logger.info(msg)


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
