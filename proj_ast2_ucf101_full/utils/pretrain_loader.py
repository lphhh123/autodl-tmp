"""Pretrained weight loader + lightweight converters.

This project intentionally uses a minimal custom ViT implementation (models/video_vit.py)
instead of timm/transformers to keep the codebase self-contained.

To support *ethical* and reproducible use of public pretrained weights, we provide:
  - a safe resolver for local weight files (no internet)
  - a converter for timm-style ViT state_dict -> our VideoViT / VideoAudioAST

Design goals:
  - never silently "do nothing" when pretrain is requested
  - never crash when pretrain is disabled
  - keep the conversion narrow and explicit (only ViT-B/16 style keys)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _cfg_get(cfg: Any, path: str, default=None):
    """Small helper for OmegaConf / dict / attr-style configs."""
    cur = cfg
    for p in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return default if cur is None else cur


def _find_weight_file(p: Path) -> Path:
    """Resolve a weight file from a directory or file path."""
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"[PRETRAIN] path not found: {p}")
    if not p.is_dir():
        raise FileNotFoundError(f"[PRETRAIN] not a file/dir: {p}")

    # Common HF snapshot layouts.
    # NOTE: many training servers are offline and may not have safetensors installed.
    # Prefer .bin/.pth when available; fall back to safetensors.
    candidates = [
        "pytorch_model.bin",
        "model.bin",
        "pytorch_model.pth",
        "model.pth",
        "checkpoint.pth",
        "weights.pth",
        "model.pt",
        "weights.pt",
        "model.safetensors",
        "pytorch_model.safetensors",
    ]
    for name in candidates:
        f = p / name
        if f.exists() and f.is_file():
            return f

    for ext in (".bin", ".pth", ".pt", ".safetensors"):
        hits = sorted(p.glob(f"*{ext}"))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"[PRETRAIN] could not find weights in {p}. Expected one of {candidates} or any *.bin/*.pth/*.pt/*.safetensors"
    )


def _load_raw_state_dict(weight_file: Path) -> Dict[str, torch.Tensor]:
    """Load a state_dict-like mapping from a local file."""
    if weight_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "[PRETRAIN] safetensors file detected but safetensors is not available. "
                "Install safetensors or provide a .pth/.pt/.bin file."
            ) from e
        return load_file(str(weight_file))

    obj = torch.load(str(weight_file), map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    raise RuntimeError(f"[PRETRAIN] unsupported weight file payload type: {type(obj)}")


def _looks_like_timm_vit(sd: Dict[str, torch.Tensor]) -> bool:
    return ("patch_embed.proj.weight" in sd) and ("blocks.0.attn.qkv.weight" in sd)


def _convert_timm_vit_to_custom_videovit(
    *,
    sd: Dict[str, torch.Tensor],
    depth: int,
    embed_dim: int,
    mlp_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Convert timm ViT weights to our custom VideoViT / VideoAudioAST keys."""
    out: Dict[str, torch.Tensor] = {}

    w = sd.get("patch_embed.proj.weight", None)
    b = sd.get("patch_embed.proj.bias", None)
    if w is None:
        raise KeyError("[PRETRAIN] missing key patch_embed.proj.weight")
    if w.ndim == 4:
        out["patch_embed.weight"] = w.reshape(w.shape[0], -1).contiguous()
    else:
        out["patch_embed.weight"] = w.contiguous()
    if b is not None:
        out["patch_embed.bias"] = b.contiguous()

    for k in ("cls_token", "pos_embed"):
        if k in sd:
            out[k] = sd[k].contiguous()

    hidden = int(embed_dim * mlp_ratio)
    for i in range(depth):
        for n in ("norm1", "norm2"):
            for t in ("weight", "bias"):
                src = f"blocks.{i}.{n}.{t}"
                if src in sd:
                    out[src] = sd[src].contiguous()

        qkv_w = sd.get(f"blocks.{i}.attn.qkv.weight", None)
        qkv_b = sd.get(f"blocks.{i}.attn.qkv.bias", None)
        proj_w = sd.get(f"blocks.{i}.attn.proj.weight", None)
        proj_b = sd.get(f"blocks.{i}.attn.proj.bias", None)
        if qkv_w is not None:
            out[f"blocks.{i}.attn.in_proj_weight"] = qkv_w.contiguous()
        if qkv_b is not None:
            out[f"blocks.{i}.attn.in_proj_bias"] = qkv_b.contiguous()
        if proj_w is not None:
            out[f"blocks.{i}.attn.out_proj.weight"] = proj_w.contiguous()
        if proj_b is not None:
            out[f"blocks.{i}.attn.out_proj.bias"] = proj_b.contiguous()

        fc1_w = sd.get(f"blocks.{i}.mlp.fc1.weight", None)
        fc1_b = sd.get(f"blocks.{i}.mlp.fc1.bias", None)
        fc2_w = sd.get(f"blocks.{i}.mlp.fc2.weight", None)
        fc2_b = sd.get(f"blocks.{i}.mlp.fc2.bias", None)
        if fc1_w is not None and fc1_w.shape[0] == hidden:
            out[f"blocks.{i}.mlp.fc1.weight"] = fc1_w.contiguous()
        if fc1_b is not None and fc1_b.numel() == hidden:
            out[f"blocks.{i}.mlp.fc1.bias"] = fc1_b.contiguous()
        if fc2_w is not None and fc2_w.shape[1] == hidden:
            out[f"blocks.{i}.mlp.fc2.weight"] = fc2_w.contiguous()
        if fc2_b is not None and fc2_b.numel() == embed_dim:
            out[f"blocks.{i}.mlp.fc2.bias"] = fc2_b.contiguous()

    for t in ("weight", "bias"):
        k = f"norm.{t}"
        if k in sd:
            out[k] = sd[k].contiguous()

    return out


def maybe_load_pretrained(*, cfg: Any, model: torch.nn.Module, logger=None) -> Tuple[bool, Dict[str, Any]]:
    pre = _cfg_get(cfg, "pretrain", None)
    enabled = _as_bool(_cfg_get(pre, "enabled", None), default=False) or _as_bool(os.environ.get("USE_PRETRAINED", "0"))
    if not enabled:
        return False, {"enabled": False}

    kind = str(_cfg_get(pre, "kind", None) or os.environ.get("PRETRAIN_KIND", "timm_vit")).strip().lower()
    path_raw = _cfg_get(pre, "path", None) or os.environ.get("PRETRAIN_PATH", None)

    if path_raw is None:
        base_dir = os.environ.get("PRETRAIN_DIR", None)
        if base_dir is not None:
            path_raw = str(Path(base_dir) / "vit_base_patch16_224.augreg_in21k_ft_in1k")

    if path_raw is None:
        raise RuntimeError("[PRETRAIN] enabled but no pretrain.path / PRETRAIN_PATH / PRETRAIN_DIR provided")

    p = Path(str(path_raw)).expanduser()
    weight_file = _find_weight_file(p)
    raw_sd = _load_raw_state_dict(weight_file)

    depth = int(_cfg_get(cfg, "model.depth", 12))
    embed_dim = int(_cfg_get(cfg, "model.embed_dim", 768))
    mlp_ratio = float(_cfg_get(cfg, "model.mlp_ratio", 4.0))

    if kind in ("timm_vit", "vit_timm"):
        if not _looks_like_timm_vit(raw_sd):
            raise RuntimeError(
                f"[PRETRAIN] kind={kind} but checkpoint not timm-ViT style. example keys: {list(raw_sd.keys())[:10]}"
            )
        mapped = _convert_timm_vit_to_custom_videovit(sd=raw_sd, depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio)
    else:
        mapped = raw_sd

    incompatible = model.load_state_dict(mapped, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    min_loaded = int(_cfg_get(pre, "min_loaded_keys", 50) or 50)
    loaded_keys = len(mapped) - len(unexpected)
    if loaded_keys < min_loaded:
        raise RuntimeError(
            f"[PRETRAIN] loaded too few keys ({loaded_keys} < {min_loaded}). checkpoint likely incompatible."
        )

    msg = (
        f"[PRETRAIN] loaded kind={kind} from {weight_file} | mapped={len(mapped)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    return True, {
        "enabled": True,
        "kind": kind,
        "path": str(p),
        "weight_file": str(weight_file),
        "mapped_keys": len(mapped),
        "missing": missing,
        "unexpected": unexpected,
    }
