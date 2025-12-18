import yaml
from dataclasses import dataclass
from typing import Any, Dict

class Config(dict):
    """Dict with attribute-style access: cfg.key instead of cfg['key']."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict):
            value = Config(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config object."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data or {})


def merge_configs(*cfgs: Config) -> Config:
    """Recursively merge multiple configs (later overrides earlier)."""
    def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out
    base: Dict[str, Any] = {}
    for c in cfgs:
        base = _merge(base, dict(c))
    return Config(base)
