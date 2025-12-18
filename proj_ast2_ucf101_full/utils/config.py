
import os
from typing import Any, Dict

import yaml


class AttrDict(dict):
    """Dictionary with attribute-style access: d.key -> d['key'].

    This is intentionally lightweight to keep things simple.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(item) from e


def _to_attr(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(v) for v in obj]
    return obj


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> AttrDict:
    """Load a YAML config file into an AttrDict, supporting `_base_` includes."""
    path = os.path.expanduser(path)
    cfg = _load_yaml(path)
    if "_base_" in cfg and cfg["_base_"] is not None:
        base_path = os.path.join(os.path.dirname(path), cfg["_base_"])
        base_cfg = _load_yaml(base_path)
        cfg.pop("_base_")
        cfg = _merge_dict(base_cfg, cfg)
    return _to_attr(cfg)
