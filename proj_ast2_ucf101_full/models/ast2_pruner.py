"""AST2.0-v2 pruning modules (SPEC_version_c_v2 §4)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility functions (SPEC §4.1, §4.3, §4.4, §4.7)
# -----------------------------------------------------------------------------


def get_patch_coords(H_p: int, W_p: int, device=None) -> torch.Tensor:
    coords = []
    for i in range(H_p):
        for j in range(W_p):
            u = (i + 0.5) / H_p
            v = (j + 0.5) / W_p
            coords.append([u, v])
    return torch.tensor(coords, dtype=torch.float32, device=device)


def init_voronoi_centers(coords: torch.Tensor, num_regions: int, jitter: bool = False) -> torch.Tensor:
    N = coords.shape[0]
    idx = torch.randperm(N)[:num_regions]
    centers = coords[idx].clone()
    if jitter:
        centers = centers + 0.02 * torch.randn_like(centers)
        centers = centers.clamp(0.0, 1.0)
    return centers


def assign_voronoi_regions(coords: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # coords: [N,2], centers: [R,2]
    # distance matrix [N,R]
    dists = torch.cdist(coords, centers)
    region_ids = torch.argmin(dists, dim=1)
    return region_ids


def conv1d_mean_over_time(p_bt: torch.Tensor, L: int) -> torch.Tensor:
    # p_bt: [B*N, C, T]
    weight = torch.ones(1, 1, L, device=p_bt.device, dtype=p_bt.dtype) / float(L)
    q = F.conv1d(p_bt, weight, stride=1, padding=0, groups=1)
    return q


def compute_multi_scale_temporal_entropy(x: torch.Tensor, window_sizes: List[int], tau: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    B, T, N, C = x.shape
    p = torch.softmax(x / tau, dim=-1)
    H_list = []
    p_bt = p.permute(0, 2, 3, 1).reshape(B * N, C, T)
    for L in window_sizes:
        if L > T:
            continue
        q = conv1d_mean_over_time(p_bt, L)  # [B*N, C, T-L+1]
        T_prime = q.shape[-1]
        q_bnct = q.reshape(B, N, C, T_prime).permute(0, 3, 1, 2)  # [B,T',N,C]
        H_center = - (q_bnct * (q_bnct + eps).log()).sum(dim=-1)  # [B,T',N]
        H_flat = H_center.permute(0, 2, 1).reshape(B * N, 1, T_prime)
        H_interp = F.interpolate(H_flat, size=T, mode="linear", align_corners=False)
        H_full = H_interp.reshape(B, N, T).permute(0, 2, 1)  # [B,T,N]
        H_list.append(H_full)
    if len(H_list) == 0:
        return torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
    H_time_ms = sum(H_list) / len(H_list)
    return H_time_ms


def compute_multi_scale_spatial_entropy(
    x: torch.Tensor,
    region_ids_coarse: torch.Tensor,
    region_ids_fine: torch.Tensor,
    num_regions_coarse: int,
    num_regions_fine: int,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = torch.softmax(x / tau, dim=-1)  # [B,T,N,C]
    B, T, N, C = p.shape
    R_c = num_regions_coarse
    R_f = num_regions_fine
    p_region_coarse = x.new_zeros(B, T, R_c, C)
    p_region_fine = x.new_zeros(B, T, R_f, C)
    for r in range(R_c):
        mask_n = (region_ids_coarse == r).to(p.dtype)
        if mask_n.sum() == 0:
            continue
        mask = mask_n.view(1, 1, N, 1)
        sel = p * mask
        p_sum = sel.sum(dim=2)
        cnt = mask_n.sum().item()
        p_region_coarse[:, :, r, :] = p_sum / (cnt + eps)
    for r in range(R_f):
        mask_n = (region_ids_fine == r).to(p.dtype)
        if mask_n.sum() == 0:
            continue
        mask = mask_n.view(1, 1, N, 1)
        sel = p * mask
        p_sum = sel.sum(dim=2)
        cnt = mask_n.sum().item()
        p_region_fine[:, :, r, :] = p_sum / (cnt + eps)
    H_region_coarse = - (p_region_coarse * (p_region_coarse + eps).log()).sum(dim=-1)
    H_region_fine = - (p_region_fine * (p_region_fine + eps).log()).sum(dim=-1)
    return H_region_coarse, H_region_fine


def build_soft_token_mask(score: torch.Tensor, rho: float, temperature: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    B, T, N = score.shape
    mask_list = []
    for b in range(B):
        flat = score[b].reshape(-1)
        k = max(1, int(rho * flat.numel()))
        kth = torch.kthvalue(-flat, k).values * -1
        mask_b = torch.sigmoid((score[b] - kth) / temperature)
        mask_list.append(mask_b)
    return torch.stack(mask_list, dim=0)


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


# -----------------------------------------------------------------------------
# ASTPruner
# -----------------------------------------------------------------------------


class ASTPruner(nn.Module):
    """AST2.0-v2 pruning module (SPEC §4)."""

    def __init__(self, cfg: Any, embed_dim: int, num_heads: int, depth: int, H_p: int, W_p: int):
        super().__init__()
        self.cfg = cfg
        self.H_p = H_p
        self.W_p = W_p
        num_tokens = H_p * W_p
        coords = get_patch_coords(H_p, W_p)
        self.register_buffer("patch_coords", coords, persistent=False)
        self.num_regions_coarse = cfg.get("num_regions_coarse", 4)
        self.num_regions_fine = cfg.get("num_regions_fine", 8)
        self.centers_coarse = nn.Parameter(init_voronoi_centers(coords, self.num_regions_coarse, jitter=False))
        self.centers_fine = nn.Parameter(init_voronoi_centers(coords, self.num_regions_fine, jitter=True))

        hidden_dim = int(embed_dim * cfg.get("mlp_ratio", 4.0)) if "mlp_ratio" in cfg else int(embed_dim * 4)
        self.g_head = nn.Parameter(torch.zeros(depth, num_heads))
        self.g_ch = nn.Parameter(torch.zeros(depth, hidden_dim))
        self.g_block = nn.Parameter(torch.zeros(depth))

    # SPEC §4.6
    def _normalize(self, H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B = H.shape[0]
        H_flat = H.view(B, -1)
        H_min = H_flat.min(dim=-1, keepdim=True)[0].view(B, 1, 1)
        H_max = H_flat.max(dim=-1, keepdim=True)[0].view(B, 1, 1)
        return (H - H_min) / (H_max - H_min + eps)

    def compute_spatiotemporal_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg_ast = self.cfg
        H_time_ms = compute_multi_scale_temporal_entropy(
            x,
            cfg_ast.get("time_window_sizes", [1, 2, 4]),
            tau=cfg_ast.get("entropy_tau", 1.0),
        )  # [B,T,N]
        region_ids_coarse = assign_voronoi_regions(self.patch_coords, self.centers_coarse)
        region_ids_fine = assign_voronoi_regions(self.patch_coords, self.centers_fine)
        H_region_coarse, H_region_fine = compute_multi_scale_spatial_entropy(
            x,
            region_ids_coarse,
            region_ids_fine,
            self.num_regions_coarse,
            self.num_regions_fine,
            tau=cfg_ast.get("entropy_tau", 1.0),
        )
        B, T, N, _ = x.shape
        Ht_norm = self._normalize(H_time_ms)
        Hc_norm = self._normalize(H_region_coarse)
        Hf_norm = self._normalize(H_region_fine)
        Hc_per_token = x.new_zeros(B, T, N)
        Hf_per_token = x.new_zeros(B, T, N)
        for r in range(self.num_regions_coarse):
            mask_n = region_ids_coarse == r
            if mask_n.sum() == 0:
                continue
            Hc_per_token[:, :, mask_n] = Hc_norm[:, :, r : r + 1]
        for r in range(self.num_regions_fine):
            mask_n = region_ids_fine == r
            if mask_n.sum() == 0:
                continue
            Hf_per_token[:, :, mask_n] = Hf_norm[:, :, r : r + 1]
        score = (
            cfg_ast.get("alpha_time", 1.0) * Ht_norm
            + cfg_ast.get("beta_space_coarse", 0.5) * Hc_per_token
            + cfg_ast.get("gamma_space_fine", 0.5) * Hf_per_token
        )
        return score, {
            "Ht": Ht_norm,
            "Hc": Hc_norm,
            "Hf": Hf_norm,
            "region_ids_coarse": region_ids_coarse,
            "region_ids_fine": region_ids_fine,
        }

    def forward_token_gating(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        score, _ = self.compute_spatiotemporal_scores(x)
        mask_soft = build_soft_token_mask(
            score,
            rho=self.cfg.get("rho_token_target", 0.5),
            temperature=self.cfg.get("token_temperature", 0.1),
        )
        x_gated = x * mask_soft.unsqueeze(-1)
        sparsity_token = 1.0 - mask_soft.mean()
        return x_gated, {"sparsity_token": sparsity_token, "mask": mask_soft}

    def forward(self, token_feat: torch.Tensor) -> ASTOutputs:
        # token_feat: [B,T,N,C]
        x_gated, stats_token = self.forward_token_gating(token_feat)
        # head/channel/block gates
        head_weights = torch.sigmoid(self.g_head)
        ch_weights = torch.sigmoid(self.g_ch)
        block_weights = torch.sigmoid(self.g_block)
        sparsity_head = 1.0 - head_weights.mean()
        sparsity_ch = 1.0 - ch_weights.mean()
        sparsity_block = 1.0 - block_weights.mean()
        L_AST = (
            self.cfg.get("lambda_token", 0.0) * stats_token["sparsity_token"]
            + self.cfg.get("lambda_head", 0.0) * sparsity_head
            + self.cfg.get("lambda_ch", 0.0) * sparsity_ch
            + self.cfg.get("lambda_block", 0.0) * sparsity_block
        )
        return ASTOutputs(
            token_mask=stats_token["mask"],
            head_weights=head_weights,
            ch_weights=ch_weights,
            block_weights=block_weights,
            sparsity={
                "token": stats_token["sparsity_token"],
                "head": sparsity_head,
                "ch": sparsity_ch,
                "block": sparsity_block,
            },
            L_AST=L_AST,
        )
