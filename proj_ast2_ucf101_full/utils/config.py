import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Set

from omegaconf import OmegaConf


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError:
            raise AttributeError(item)
        return v

    def __setattr__(self, key, value):
        self[key] = value


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _to_attr(d: Any) -> Any:
    if isinstance(d, dict):
        return AttrDict({k: _to_attr(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attr(x) for x in d]
    return d


def _to_plain(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _to_plain(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_to_plain(x) for x in d]
    return d


def _load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_base_path(child_cfg_path: str, base_value: str) -> str:
    """
    Resolve _base_ robustly:
      1) absolute path
      2) relative to current cfg dir
      3) relative to repo root (utils/..)
      4) special-case: base starts with 'configs/' to avoid 'configs/configs/'
    """
    base = Path(os.path.expanduser(str(base_value)))
    if base.is_absolute():
        return str(base)

    child_dir = Path(os.path.expanduser(child_cfg_path)).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]

    cand1 = (child_dir / base).resolve()
    cand2 = (repo_root / base).resolve()

    cands = [cand1, cand2]

    if len(base.parts) > 0 and base.parts[0] == "configs":
        cand3 = (repo_root / Path(*base.parts[1:])).resolve()
        cands.insert(0, cand3)

    for c in cands:
        if c.exists():
            return str(c)

    return str(cand1)


def _load_config_dict(path: str, _seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    if _seen is None:
        _seen = set()

    path = os.path.expanduser(path)
    abs_path = str(Path(path).resolve())
    if abs_path in _seen:
        raise ValueError(f"[config] cyclic _base_ include detected: {abs_path}")
    _seen.add(abs_path)

    cfg = _load_yaml(abs_path)

    base = cfg.get("_base_")
    if base:
        base_path = _resolve_base_path(abs_path, base)
        base_cfg = _load_config_dict(base_path, _seen=_seen)
        cfg.pop("_base_", None)
        merged = _merge_dict(base_cfg, cfg)
        return merged

    return cfg


def load_config(path: str):
    """
    Return OmegaConf DictConfig (NOT AttrDict), so OmegaConf.select/to_yaml/to_container work.
    Supports recursive _base_ merge.
    """
    d = _load_config_dict(path)
    d = _to_plain(d)
    return OmegaConf.create(d)
