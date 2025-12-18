"""AST2.0-lite v2 pruning modules (SPEC_version_c_full §4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _softmax_over_channels(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(x / tau, dim=-1)


def _entropy_from_prob(p: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    return -torch.sum(p * (p + eps).log(), dim=dim)


def compute_multi_level_time_entropy(x: torch.Tensor, levels: List[int], tau: float, eps: float) -> Dict[str, torch.Tensor]:
    """Multi-level time entropy (SPEC_version_c_full §4.2.2)."""
    B, T, N, C = x.shape
    p = _softmax_over_channels(x, tau)
    results: Dict[str, torch.Tensor] = {"H_time_level": {}}
    window_cache = {}
    for L in levels:
        if L <= 0:
            L = 1
        window = max(1, int(torch.ceil(torch.tensor(T / L)).item()))
        p_w = []
        for w in range(L):
            t_start = w * window
            t_end = min((w + 1) * window, T)
            if t_start >= t_end:
                continue
            key = (t_start, t_end)
            if key not in window_cache:
                p_window = p[:, t_start:t_end].mean(dim=1)
                window_cache[key] = p_window
            p_w.append(window_cache[key])
        if len(p_w) == 0:
            continue
        p_stack = torch.stack(p_w, dim=-1)  # [B,N,C,L]
        H_w = _entropy_from_prob(p_stack, dim=2)  # [B,N,L]
        H_level = H_w.mean(dim=-1)  # [B,N]
        results["H_time_level"][L] = H_level
        if L == 1:
            results["H_time_global"] = H_level
    if "H_time_global" not in results and results["H_time_level"]:
        first = next(iter(results["H_time_level"].values()))
        results["H_time_global"] = first
    return results


def compute_multi_level_space_entropy(x: torch.Tensor, H_p: int, W_p: int, levels: List[int], tau: float, eps: float) -> Dict[str, torch.Tensor]:
    """Multi-level space entropy (SPEC_version_c_full §4.2.3)."""
    B, T, N, C = x.shape
    p = _softmax_over_channels(x, tau)
    p_reshaped = p.view(B, T, H_p, W_p, C)
    results: Dict[str, torch.Tensor] = {"H_space_level": {}}
    for L in levels:
        if L <= 0:
            L = 1
        block_h = int(torch.ceil(torch.tensor(H_p / L)).item())
        block_w = int(torch.ceil(torch.tensor(W_p / L)).item())
        H_blocks = []
        for u in range(L):
            for v in range(L):
                h_start = u * block_h
                h_end = min((u + 1) * block_h, H_p)
                w_start = v * block_w
                w_end = min((v + 1) * block_w, W_p)
                if h_start >= h_end or w_start >= w_end:
                    continue
                p_block = p_reshaped[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))
                H_block = _entropy_from_prob(p_block, dim=-1)  # [B,T]
                H_blocks.append(H_block)
        if len(H_blocks) == 0:
            continue
        H_level = torch.stack(H_blocks, dim=-1).mean(dim=-1)  # [B,T]
        results["H_space_level"][L] = H_level
        if L == 1:
            results["H_space_global"] = H_level
    if "H_space_global" not in results and results["H_space_level"]:
        results["H_space_global"] = next(iter(results["H_space_level"].values()))
    return results


def build_voronoi_regions(num_patches: int, grid_hw: Tuple[int, int], num_regions: int) -> Tuple[torch.Tensor, int]:
    """Build static Voronoi-style regions on patch grid (SPEC_version_c_full §4.3)."""
    H_p, W_p = grid_hw
    coords = []
    for i in range(H_p):
        for j in range(W_p):
            coords.append([i, j])
    coords = torch.tensor(coords, dtype=torch.float32)
    # regular grid seeds
    side = int(torch.ceil(torch.tensor(num_regions ** 0.5)).item())
    seeds = []
    for i in range(side):
        for j in range(side):
            if len(seeds) >= num_regions:
                break
            seeds.append([(i + 0.5) * H_p / side, (j + 0.5) * W_p / side])
    centers = torch.tensor(seeds, dtype=torch.float32)
    dists = torch.cdist(coords, centers)
    region_ids = torch.argmin(dists, dim=1)
    return region_ids, centers.shape[0]


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class ASTOutputs:
    token_mask: torch.Tensor
    head_weights: torch.Tensor
    ch_weights: torch.Tensor
    block_weights: torch.Tensor
    sparsity: Dict[str, torch.Tensor]
    L_AST: torch.Tensor
    extras: Dict[str, Any]


# -----------------------------------------------------------------------------
# ASTPruner
# -----------------------------------------------------------------------------


class ASTPruner(nn.Module):
    """AST2.0-lite v2 pruning (SPEC_version_c_full §4)."""

    def __init__(self, cfg: Any, embed_dim: int, num_heads: int, depth: int, H_p: int, W_p: int):
        super().__init__()
        self.cfg = cfg
        self.H_p = H_p
        self.W_p = W_p
        self.num_patches = H_p * W_p
        self.time_levels = cfg.get("time_window_levels", [1, 2, 4])
        self.space_levels = cfg.get("space_window_levels", [1, 2, 4])
        self.entropy_tau = cfg.get("entropy_tau", 1.0)
        self.entropy_eps = 1e-6
        region_ids, num_regions = build_voronoi_regions(self.num_patches, (H_p, W_p), cfg.get("num_voronoi_regions", cfg.get("num_regions_coarse", 4)))
        self.register_buffer("region_ids", region_ids, persistent=False)
        self.num_regions = num_regions

        hidden_dim = int(embed_dim * cfg.get("mlp_ratio", 4.0)) if "mlp_ratio" in cfg else int(embed_dim * 4)
        self.head_logit = nn.Parameter(torch.zeros(depth, num_heads))
        self.ch_logit = nn.Parameter(torch.zeros(depth, hidden_dim))
        self.block_logit = nn.Parameter(torch.zeros(depth))

    # SPEC §4.6
    def _normalize(self, H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B = H.shape[0]
        H_flat = H.view(B, -1)
        H_min = H_flat.min(dim=-1, keepdim=True)[0].view(B, 1, 1)
        H_max = H_flat.max(dim=-1, keepdim=True)[0].view(B, 1, 1)
        return (H - H_min) / (H_max - H_min + eps)

    def _fuse_time_entropy(self, H_time_levels: Dict[str, torch.Tensor]) -> torch.Tensor:
        H_list = []
        for _, H in H_time_levels.get("H_time_level", {}).items():
            H_list.append(self._normalize(H))
        if not H_list:
            return torch.zeros_like(H_time_levels.get("H_time_global"))
        return sum(H_list) / len(H_list)

    def _fuse_space_entropy(self, H_space_levels: Dict[str, torch.Tensor]) -> torch.Tensor:
        H_list = []
        for _, H in H_space_levels.get("H_space_level", {}).items():
            H_list.append(self._normalize(H))
        if not H_list:
            return torch.zeros_like(H_space_levels.get("H_space_global"))
        return sum(H_list) / len(H_list)

    def token_gating(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg_ast = self.cfg
        H_time_levels = compute_multi_level_time_entropy(x, self.time_levels, self.entropy_tau, self.entropy_eps)
        H_space_levels = compute_multi_level_space_entropy(x, self.H_p, self.W_p, self.space_levels, self.entropy_tau, self.entropy_eps)
        H_time_fused = self._fuse_time_entropy(H_time_levels)  # [B,N]
        H_space_fused = self._fuse_space_entropy(H_space_levels)  # [B,T]
        region_ids = self.region_ids.to(x.device)
        B, T, N, _ = x.shape
        region_importance = []
        for r in range(self.num_regions):
            mask = (region_ids == r).float()
            if mask.sum() == 0:
                region_importance.append(torch.zeros(B, device=x.device, dtype=x.dtype))
                continue
            region_importance.append(H_space_fused.mean(dim=1))
        region_importance = torch.stack(region_importance, dim=1)  # [B,R]
        score = x.new_zeros(B, T, N)
        for n in range(N):
            r = int(region_ids[n].item())
            score[:, :, n] = (
                cfg_ast.get("a_time", cfg_ast.get("alpha_time", 1.0)) * H_time_fused[:, n].unsqueeze(1)
                + cfg_ast.get("b_space", cfg_ast.get("beta_space_coarse", 0.5)) * H_space_fused
                + cfg_ast.get("c_region", cfg_ast.get("gamma_space_fine", 0.5)) * region_importance[:, r].unsqueeze(1)
            )
        mask_list = []
        rho = cfg_ast.get("rho_token_target", 0.5)
        temperature = cfg_ast.get("token_temperature", 0.1)
        for b in range(B):
            flat = score[b].reshape(-1)
            k = max(1, int(rho * flat.numel()))
            threshold = torch.kthvalue(-flat, k).values * -1
            mask_b = torch.sigmoid((score[b] - threshold) / temperature)
            mask_list.append(mask_b)
        mask = torch.stack(mask_list, dim=0)
        sparsity_token = 1.0 - mask.mean()
        return mask, {
            "sparsity_token": sparsity_token,
            "H_time": H_time_levels,
            "H_space": H_space_levels,
            "region_importance": region_importance,
        }

    def forward(self, token_feat: torch.Tensor) -> ASTOutputs:
        # token_feat: [B,T,N,C]
        mask, token_stats = self.token_gating(token_feat)
        x_gated = token_feat * mask.unsqueeze(-1)
        head_weights = torch.sigmoid(self.head_logit)
        ch_weights = torch.sigmoid(self.ch_logit)
        block_weights = torch.sigmoid(self.block_logit)
        sparsity_head = 1.0 - head_weights.mean()
        sparsity_ch = 1.0 - ch_weights.mean()
        sparsity_block = 1.0 - block_weights.mean()
        L_AST = (
            self.cfg.get("lambda_token", 0.0) * token_stats["sparsity_token"]
            + self.cfg.get("lambda_head", 0.0) * sparsity_head
            + self.cfg.get("lambda_ch", 0.0) * sparsity_ch
            + self.cfg.get("lambda_block", 0.0) * sparsity_block
        )
        return ASTOutputs(
            token_mask=mask,
            head_weights=head_weights,
            ch_weights=ch_weights,
            block_weights=block_weights,
            sparsity={
                "token": token_stats["sparsity_token"],
                "head": sparsity_head,
                "ch": sparsity_ch,
                "block": sparsity_block,
            },
            L_AST=L_AST,
            extras=token_stats,
        )
