from __future__ import annotations

from pathlib import Path
from typing import Any

FILE_LIST_FULL = {
    "budget.json",
    "checkpoint_state.json",
    "config_requested.yaml",
    "effective_config_snapshot.yaml",
    "eval_config_snapshot.yaml",
    "layout_best.json",
    "manifest.json",
    "pareto_points.csv",
    "report.json",
    "requested_config_snapshot.yaml",
    "summary.json",
    "trace.csv",
    "trace_header.json",
    "trace_meta.json",
}

DEFAULT_INCLUDE_EXTS = {".json", ".yaml", ".yml", ".csv", ".jsonl", ".flag", ".txt"}

FILTER_FRAMES = ("utils/io_compat/", "utils\\io_compat\\")


class CompatSettings:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        project_root = Path(__file__).resolve().parents[2]
        default_log = project_root / ".io_compat_write.jsonl"
        self.log_path: Path = Path(cfg.get("log_path", default_log)).resolve()
        self.record_all_writes: bool = bool(cfg.get("record_all_writes", False))
        self.include_exts: set[str] = {
            str(ext).lower() for ext in cfg.get("include_exts", DEFAULT_INCLUDE_EXTS)
        }
        self.file_list_full: set[str] = set(cfg.get("file_list_full", FILE_LIST_FULL))


SETTINGS = CompatSettings()


def update_settings(config: dict[str, Any] | None = None) -> CompatSettings:
    global SETTINGS
    SETTINGS = CompatSettings(config)
    return SETTINGS
