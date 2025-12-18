from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def token_channel_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy along the channel dimension.

    Args:
        x: [B, L, C] token features

    Returns:
        entropy: [B, L]
    """
    # Normalize along channels and compute Shannon entropy
    # Small temperature to avoid over-smoothing
    p = F.softmax(x, dim=-1)
    entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
    return entropy  # [B, L]


def build_token_coords(num_frames: int,
                       num_patches: int,
                       height_patches: int,
                       width_patches: int,
                       device: torch.device) -> torch.Tensor:
    """Build (t, y, x) coordinates for each token.

    Returns:
        coords: [L, 3] with normalized coordinates in [0, 1].
    """
    assert height_patches * width_patches == num_patches
    coords = []
    for t in range(num_frames):
        for h in range(height_patches):
            for w in range(width_patches):
                coords.append([
                    t / max(num_frames - 1, 1),
                    h / max(height_patches - 1, 1),
                    w / max(width_patches - 1, 1),
                ])
    return torch.tensor(coords, dtype=torch.float32, device=device)  # [L, 3]


def entropy_voronoi_importance(
    x: torch.Tensor,
    num_frames: int,
    height_patches: int,
    width_patches: int,
    num_centers: int = 16,
    sigma: float = 0.15,
) -> torch.Tensor:
    """Compute entropy * Voronoi-based geometric weighting as importance.

    This is a simplified, differentiable approximation of the
    "spatio-temporal entropy + Voronoi sampling" idea in AST2.0.

    Args:
        x: token features [B, L, C] where L = num_frames * num_patches
        num_frames, height_patches, width_patches: geometry
        num_centers: number of Voronoi centers used
        sigma: distance scale for Gaussian weighting

    Returns:
        importance: [B, L] larger = more important
    """
    B, L, C = x.shape
    device = x.device
    num_patches = height_patches * width_patches
    assert L == num_frames * num_patches

    entropy = token_channel_entropy(x)  # [B, L]

    coords = build_token_coords(num_frames, num_patches,
                                height_patches, width_patches, device)  # [L, 3]

    # Pick centers uniformly in the grid (deterministic)
    if num_centers >= L:
        centers = coords
    else:
        # Simple uniform sampling over frames and spatial grid
        idx_per_frame = max(1, num_centers // max(num_frames, 1))
        indices = []
        for t in range(num_frames):
            base = t * num_patches
            step = max(1, num_patches // idx_per_frame)
            for i in range(0, num_patches, step):
                indices.append(base + i)
                if len(indices) >= num_centers:
                    break
            if len(indices) >= num_centers:
                break
        centers = coords[indices]  # [K, 3]

    # Distances from tokens to centers
    d2 = torch.cdist(coords, centers) ** 2  # [L, K]
    # use distance to nearest center
    nearest_d2, _ = d2.min(dim=-1)  # [L]
    weights = torch.exp(-nearest_d2 / (2 * sigma ** 2))  # [L]
    weights = weights.clamp_min(1e-4)

    # Broadcast to batches
    weights = weights.unsqueeze(0).expand(B, -1)  # [B, L]

    importance = entropy * weights
    return importance


def topk_mask_from_importance(
    importance: torch.Tensor,
    keep_ratio: float,
) -> torch.Tensor:
    """Build a binary mask selecting top-k tokens per sample.

    Args:
        importance: [B, L]
        keep_ratio: in (0, 1]

    Returns:
        mask: [B, L] with {0,1}
    """
    B, L = importance.shape
    k = max(1, int(L * keep_ratio))
    # topk returns values and indices
    _, idx = importance.topk(k, dim=-1, largest=True, sorted=False)
    mask = importance.new_zeros((B, L))
    mask.scatter_(1, idx, 1.0)
    return mask


def sparsity_loss_from_keep_ratios(
    keep_ratios: List[torch.Tensor],
    target_start: float,
    target_end: float,
) -> torch.Tensor:
    """Simple L2 loss to encourage keep_ratios to follow a depth-wise schedule.

    Args:
        keep_ratios: list of scalar tensors (one per layer)
        target_start: target keep_ratio at shallowest layer
        target_end: target keep_ratio at deepest layer

    Returns:
        scalar tensor
    """
    if len(keep_ratios) == 0:
        return keep_ratios[0].new_tensor(0.0)

    L = len(keep_ratios)
    targets = torch.linspace(target_start, target_end, L,
                             device=keep_ratios[0].device)
    kr = torch.stack(keep_ratios, dim=0)
    loss = F.mse_loss(kr, targets)
    return loss


def compute_entropy_sparsity_loss(
    model: nn.Module,
    lambda_keep: float,
    target_start: float,
    target_end: float,
) -> torch.Tensor:
    """Wrapper used by trainers.

    Requires model to implement get_keep_ratios() -> List[Tensor].
    """
    if not hasattr(model, "get_keep_ratios"):
        raise AttributeError("Model must implement get_keep_ratios()")

    keep_ratios = model.get_keep_ratios()
    if len(keep_ratios) == 0:
        return keep_ratios[0].new_tensor(0.0)

    loss = sparsity_loss_from_keep_ratios(
        keep_ratios,
        target_start=target_start,
        target_end=target_end,
    )
    return lambda_keep * loss
