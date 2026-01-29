from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert common runtime objects (torch/numpy/path/etc.) into JSON-serializable types.
    Use this when you need guaranteed pure-Python containers (e.g. before writing to json or yaml).
    """
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # torch
    try:
        import torch
        if torch.is_tensor(obj):
            t = obj.detach().cpu()
            if t.numel() == 1:
                return t.item()
            return t.tolist()
    except Exception:
        pass

    # dict
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = to_jsonable(v)
        return out

    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # fallback: string repr
    return str(obj)


def _safe_default(o: Any) -> Any:
    """
    json.dump/json.dumps default handler.
    Converts torch/numpy/path/etc. without requiring callers to pre-walk the object graph.
    """
    # Path
    if isinstance(o, Path):
        return str(o)

    # numpy
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass

    # torch
    try:
        import torch
        if torch.is_tensor(o):
            t = o.detach().cpu()
            if t.numel() == 1:
                return t.item()
            return t.tolist()
    except Exception:
        pass

    # set/tuple -> list
    if isinstance(o, (set, tuple)):
        return list(o)

    # fallback
    return str(o)


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return _safe_default(o)


def safe_dumps(obj: Any, **kwargs) -> str:
    """
    Safer json.dumps that can handle torch/numpy/path/etc.
    """
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)


def safe_dump(obj: Any, fp, **kwargs) -> None:
    """
    Safer json.dump that can handle torch/numpy/path/etc.
    """
    kwargs.setdefault("ensure_ascii", False)
    json.dump(obj, fp, cls=SafeJSONEncoder, **kwargs)
