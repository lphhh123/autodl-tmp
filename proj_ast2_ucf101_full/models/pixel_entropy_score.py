"""Pixel-entropy based token importance (paper-style signal).

score(patch) = (H_local / (H_global + eps)) * exp(-lambda_dist * dist)

- H_local: patch histogram entropy (optionally fused with temporal entropy on |frame diff|)
- H_global: frame-level histogram entropy
- dist: normalized distance to an anchor (default: max-entropy patch)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_tuple3(x: Optional[Iterable[float]], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if x is None:
        return default
    xs = list(x)
    if len(xs) != 3:
        return default
    return (float(xs[0]), float(xs[1]), float(xs[2]))


def _quantize_u8_to_bins(u8: torch.Tensor, bins: int) -> torch.Tensor:
    """Map uint8 [0,255] to integer bins [0,bins-1]."""
    if bins <= 1:
        return torch.zeros_like(u8, dtype=torch.int64)
    if bins == 256:
        return u8.to(torch.int64)
    q = torch.div(u8.to(torch.int64) * int(bins), 256, rounding_mode="floor")
    return q.clamp_(0, int(bins) - 1)


def _entropy_from_quantized(vals: torch.Tensor, bins: int, eps: float) -> torch.Tensor:
    """vals: Long [M,K] in [0,bins-1] -> entropy [M]."""
    if vals.numel() == 0:
        return vals.new_zeros((vals.shape[0],), dtype=torch.float32)
    m, k = vals.shape
    hist = torch.zeros((m, int(bins)), device=vals.device, dtype=torch.float32)
    ones = torch.ones((m, k), device=vals.device, dtype=torch.float32)
    hist.scatter_add_(1, vals, ones)
    p = hist / float(k)
    return -(p * (p + float(eps)).log()).sum(dim=1)


def _minmax_norm_last(x: torch.Tensor, eps: float) -> torch.Tensor:
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    return (x - mn) / (mx - mn + float(eps))


@torch.no_grad()
def compute_pixel_entropy_density_score_video(
    x_norm: torch.Tensor,
    patch_size: int,
    *,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
    bins: int = 32,
    eps: float = 1.0e-6,
    patch_downsample: int = 2,
    use_temporal: bool = True,
    temporal_delta: int = 1,
    lambda_dist: float = 0.5,
    center_mode: str = "max_entropy",
    alpha_space: float = 1.0,
    beta_time: float = 1.0,
    normalize_per_frame: bool = True,
) -> torch.Tensor:
    assert x_norm.dim() == 5, f"Expected [B,T,C,H,W], got {tuple(x_norm.shape)}"
    b, t, c, h, w = x_norm.shape
    assert c == 3, "pixel-entropy scorer expects RGB input"
    assert h % int(patch_size) == 0 and w % int(patch_size) == 0, "H/W must be divisible by patch_size"
    hp = h // int(patch_size)
    wp = w // int(patch_size)
    n = hp * wp

    mean = _as_tuple3(mean, IMAGENET_MEAN)
    std = _as_tuple3(std, IMAGENET_STD)
    mean_t = x_norm.new_tensor(mean).view(1, 1, 3, 1, 1)
    std_t = x_norm.new_tensor(std).view(1, 1, 3, 1, 1)

    x = x_norm.float() * std_t.float() + mean_t.float()
    x = x.clamp(0.0, 1.0)
    gray = (0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]).contiguous()

    bt = b * t
    ds = int(max(1, patch_downsample))

    gray_bt = gray.view(bt, 1, h, w)
    patches = F.unfold(gray_bt, kernel_size=int(patch_size), stride=int(patch_size))
    patches = patches.transpose(1, 2).contiguous()
    if ds > 1:
        ps = int(patch_size)
        patches = patches.view(bt, n, ps, ps)[:, :, ::ds, ::ds].contiguous().view(bt, n, -1)
    p_eff = int(patches.shape[-1])

    u8 = (patches * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    q = _quantize_u8_to_bins(u8, int(bins)).view(bt * n, p_eff)
    h_space = _entropy_from_quantized(q, int(bins), float(eps)).view(b, t, n)

    gray_ds = gray[:, :, ::ds, ::ds]
    u8g = (gray_ds * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    qg = _quantize_u8_to_bins(u8g, int(bins)).view(bt, -1)
    h_global_space = _entropy_from_quantized(qg, int(bins), float(eps)).view(b, t)

    if use_temporal and t > 1:
        d = int(max(1, temporal_delta))
        if d >= t:
            d = 1
        gray_shift = torch.roll(gray, shifts=-d, dims=1)
        diff = (gray_shift - gray).abs()
        diff[:, -d:] = 0.0

        diff_bt = diff.view(bt, 1, h, w)
        diff_p = F.unfold(diff_bt, kernel_size=int(patch_size), stride=int(patch_size)).transpose(1, 2).contiguous()
        if ds > 1:
            ps = int(patch_size)
            diff_p = diff_p.view(bt, n, ps, ps)[:, :, ::ds, ::ds].contiguous().view(bt, n, -1)
        p_eff_t = int(diff_p.shape[-1])

        u8t = (diff_p * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
        qt = _quantize_u8_to_bins(u8t, int(bins)).view(bt * n, p_eff_t)
        h_time = _entropy_from_quantized(qt, int(bins), float(eps)).view(b, t, n)

        diff_ds = diff[:, :, ::ds, ::ds]
        u8gt = (diff_ds * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
        qgt = _quantize_u8_to_bins(u8gt, int(bins)).view(bt, -1)
        h_global_time = _entropy_from_quantized(qgt, int(bins), float(eps)).view(b, t)
    else:
        h_time = h_space.new_zeros((b, t, n))
        h_global_time = h_global_space.new_zeros((b, t))

    yy = torch.arange(hp, device=x_norm.device, dtype=torch.float32).repeat_interleave(wp).view(1, 1, n)
    xx = torch.arange(wp, device=x_norm.device, dtype=torch.float32).repeat(hp).view(1, 1, n)
    norm = float(math.sqrt(max(1.0, (hp - 1) ** 2 + (wp - 1) ** 2)))

    cm = str(center_mode or "max_entropy").lower()
    if cm in ("center", "image_center", "frame_center"):
        y0 = yy.new_full((b, t, 1), float(hp // 2))
        x0 = xx.new_full((b, t, 1), float(wp // 2))
    else:
        idx0 = h_space.argmax(dim=-1)
        y0 = (idx0 // int(wp)).to(torch.float32).unsqueeze(-1).to(device=x_norm.device)
        x0 = (idx0 % int(wp)).to(torch.float32).unsqueeze(-1).to(device=x_norm.device)

    dist = torch.sqrt((yy - y0) ** 2 + (xx - x0) ** 2) / (norm + float(eps))
    w_dist = torch.exp(-float(lambda_dist) * dist)

    h_local = float(alpha_space) * h_space + float(beta_time) * h_time
    h_global = float(alpha_space) * h_global_space + float(beta_time) * h_global_time
    ratio = h_local / (h_global.unsqueeze(-1) + float(eps))
    score = ratio * w_dist

    if normalize_per_frame:
        score = _minmax_norm_last(score, float(eps))

    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
