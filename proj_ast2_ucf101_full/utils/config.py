
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


def load_config(path: str) -> AttrDict:
    """Load a YAML config file into an AttrDict.

    Parameters
    ----------
    path: str
        Path to the yaml file.

    Returns
    -------
    AttrDict
        Nested dictionary with attribute-style access.
    """
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _to_attr(cfg)
