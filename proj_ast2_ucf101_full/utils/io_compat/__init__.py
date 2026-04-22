from __future__ import annotations

from typing import Any

from .patch import install_patches, uninstall_patches


def init_utils(config: dict[str, Any] | None = None) -> None:
    install_patches(config)


def cleanup_utils() -> None:
    uninstall_patches()


__all__ = ["init_utils", "cleanup_utils"]
