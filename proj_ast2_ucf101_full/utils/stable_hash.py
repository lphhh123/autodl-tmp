import hashlib
import json
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass

    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def stable_hash(obj: Any) -> str:
    jsonable = _to_jsonable(obj)
    s = json.dumps(
        jsonable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
