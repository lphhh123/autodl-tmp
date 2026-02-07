"""AST2.0-v2 pruning modules (SPEC_version_c_v2 §4)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility functions (SPEC §4.2 - §4.7)
# -----------------------------------------------------------------------------


def _softmax_over_channels(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(x / tau, dim=-1)


def _entropy_from_prob(p: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=dim)


def minmax_norm_per_batch(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    x_flat = x.view(B, -1)
    mn = x_flat.min(dim=1).values.view(B, *([1] * (x.dim() - 1)))
    mx = x_flat.max(dim=1).values.view(B, *([1] * (x.dim() - 1)))
    return (x - mn) / (mx - mn + 1e-6)


def build_voronoi_regions(num_patches: int, grid_hw: Tuple[int, int], num_regions: int) -> Tuple[torch.Tensor, int]:
    H_p, W_p = grid_hw
    coords = []
    for i in range(H_p):
        for j in range(W_p):
            coords.append([i, j])
    coords_t = torch.tensor(coords, dtype=torch.float32)
    if num_regions <= 0:
        num_regions = 1
    k = int(math.ceil(math.sqrt(num_regions)))
    seeds = []
    for i in range(k):
        for j in range(k):
            seeds.append([i * (H_p / k), j * (W_p / k)])
            if len(seeds) >= num_regions:
                break
        if len(seeds) >= num_regions:
            break
    seeds_t = torch.tensor(seeds, dtype=torch.float32)
    dists = torch.cdist(coords_t, seeds_t)
    region_ids = torch.argmin(dists, dim=1)
    if num_patches > region_ids.numel():
        pad = torch.zeros(num_patches - region_ids.numel(), dtype=region_ids.dtype)
        region_ids = torch.cat([region_ids, pad], dim=0)
    return region_ids[:num_patches], num_regions


def conv1d_mean_over_time(p_bt: torch.Tensor, L: int, stride: int = 1) -> torch.Tensor:
    # p_bt: [B*N, C, T]
    C = p_bt.shape[1]
    weight = torch.ones(C, 1, L, device=p_bt.device, dtype=p_bt.dtype) / float(L)
    q = F.conv1d(p_bt, weight, stride=stride, padding=0, groups=C)
    return q


def compute_multi_level_time_entropy(
    x: torch.Tensor,
    levels: List[int],
    tau: float,
    eps: float,
    overlap: float = 0.0,
) -> Dict[str, torch.Tensor]:
    B, T, N, C = x.shape
    p = _softmax_over_channels(x, tau)
    p_bt = p.permute(0, 2, 3, 1).reshape(B * N, C, T)
    H_time_level: Dict[int, torch.Tensor] = {}
    H_time_ms_list = []
    for L in levels:
        if L <= 0 or T < L:
            continue
        stride = max(1, int(round(L * (1.0 - overlap))))
        q = conv1d_mean_over_time(p_bt, L, stride=stride)
        T_prime = q.shape[-1]
        q_bnct = q.reshape(B, N, C, T_prime).permute(0, 3, 1, 2)
        H_L_center = _entropy_from_prob(q_bnct, dim=-1, eps=eps)
        H_flat = H_L_center.permute(0, 2, 1).reshape(B * N, 1, T_prime)
        H_interp = F.interpolate(H_flat, size=T, mode="linear", align_corners=False)
        H_L_full = H_interp.reshape(B, N, T).permute(0, 2, 1)
        H_time_ms_list.append(H_L_full)
        H_time_level[L] = H_L_full.mean(dim=1)
    if H_time_ms_list:
        H_time_ms = torch.stack(H_time_ms_list, dim=0).mean(dim=0)
    else:
        H_time_ms = torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
    return {"H_time_level": H_time_level, "H_time_ms": H_time_ms}


def compute_multi_level_space_entropy(
    x: torch.Tensor,
    H_p: int,
    W_p: int,
    levels: List[int],
    tau: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    B, T, N, C = x.shape
    p = _softmax_over_channels(x, tau)
    p_hw = p.view(B, T, H_p, W_p, C)
    out: Dict[str, torch.Tensor] = {"H_space_level": {}}
    for L in levels:
        block_h = int(math.ceil(H_p / L))
        block_w = int(math.ceil(W_p / L))
        blocks = []
        for u in range(L):
            for v in range(L):
                h_start = u * block_h
                h_end = min(H_p, h_start + block_h)
                w_start = v * block_w
                w_end = min(W_p, w_start + block_w)
                if h_start >= h_end or w_start >= w_end:
                    continue
                p_block = p_hw[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))
                H_block = _entropy_from_prob(p_block, dim=-1, eps=eps)  # [B, T]
                blocks.append(H_block)
        if not blocks:
            continue
        H_stack = torch.stack(blocks, dim=-1)  # [B, T, L*L]
        out["H_space_level"][L] = H_stack.mean(dim=-1)
    if 1 in out["H_space_level"]:
        out["H_space_global"] = out["H_space_level"][1]
    else:
        out["H_space_global"] = torch.zeros(B, T, device=x.device, dtype=x.dtype)
    return out


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
    """AST2.0-lite pruning module (multi-scale sliding window + multi-modal)."""

    def __init__(self, cfg: Any, embed_dim: int, num_heads: int, depth: int, num_patches: int):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_patches = num_patches

        self.time_window_levels = cfg.get("time_window_levels", [1, 2, 4])
        self.space_window_levels = cfg.get("space_window_levels", [1, 2, 4])
        self.time_window_overlap = cfg.get("time_window_overlap", 0.0)
        self.entropy_tau = cfg.get("entropy_tau", 1.0)
        self.entropy_eps = 1e-6

        H_p = cfg.get("patch_grid_h", 14)
        W_p = cfg.get("patch_grid_w", 14)
        num_regions = cfg.get("num_regions", 8)
        region_ids, num_regions = build_voronoi_regions(num_patches, (H_p, W_p), num_regions)
        self.register_buffer("region_ids", region_ids, persistent=False)
        self.num_regions = num_regions
        self.H_p = H_p
        self.W_p = W_p

        self.num_modalities = cfg.get("num_modalities", 1)
        self.modalities = cfg.get("modalities", ["video"])
        self.modality_logit = nn.Parameter(torch.zeros(self.num_modalities))

        hidden_dim = int(embed_dim * cfg.get("mlp_ratio", 4.0)) if "mlp_ratio" in cfg else int(embed_dim * 4)
        self.g_head = nn.Parameter(torch.zeros(depth, num_heads))
        self.g_ch = nn.Parameter(torch.zeros(depth, hidden_dim))
        self.g_block = nn.Parameter(torch.zeros(depth))

        self.gating_fp32 = bool(cfg.get("gating_fp32", True))
        self.detach_mask = bool(cfg.get("detach_mask", True))
        self.hard_topk = bool(cfg.get("hard_topk", True))
        self.min_keep_ratio = float(cfg.get("min_keep_ratio", 0.25))
        self.sanitize_nan = bool(cfg.get("sanitize_nan", True))

        # Runtime overrides (set by trainer for warmup/ramp scheduling).
        # These are optional and default to None/False to preserve original behavior.
        self._runtime_force_dense: bool = False
        self._runtime_rho_token: Optional[float] = None
        self._runtime_token_temperature: Optional[float] = None

    def set_runtime_overrides(
        self,
        *,
        force_dense: Optional[bool] = None,
        rho_token: Optional[float] = None,
        token_temperature: Optional[float] = None,
    ) -> None:
        """Set per-epoch runtime overrides without mutating cfg."""
        if force_dense is not None:
            self._runtime_force_dense = bool(force_dense)
        if rho_token is not None:
            self._runtime_rho_token = float(rho_token)
        else:
            self._runtime_rho_token = None
        if token_temperature is not None:
            self._runtime_token_temperature = float(token_temperature)
        else:
            self._runtime_token_temperature = None

    def _normalize(self, H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return minmax_norm_per_batch(H)

    @staticmethod
    def _calibrate_keep_mean(
        mask_soft: torch.Tensor,
        target_keep: float,
        iters: int = 2,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Calibrate a soft mask so its per-sample mean matches target_keep (keep ratio).
        This keeps sparsity tracking rho even when sigmoid temperature is high or
        gate-score scale is small.

        mask_soft shape: [B, T, N], values in [0, 1].
        """
        target_keep = float(target_keep)
        if target_keep >= 1.0 - 1e-6:
            return torch.ones_like(mask_soft)
        if target_keep <= 0.0 + 1e-6:
            return torch.zeros_like(mask_soft)

        out = mask_soft
        for _ in range(max(1, int(iters))):
            cur = out.mean(dim=(1, 2), keepdim=True)
            scale = target_keep / (cur + eps)
            out = (out * scale).clamp(0.0, 1.0)
        return out

    def _fuse_levels(
        self,
        levels: Dict[int, torch.Tensor],
        expected_shape: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        fused = []
        for H in levels.values():
            if expected_shape is not None:
                assert tuple(H.shape) == tuple(expected_shape), (
                    f"Expected level shape {expected_shape}, got {tuple(H.shape)}"
                )
            fused.append(self._normalize(H))
        if not fused:
            if expected_shape is None:
                raise ValueError("Expected shape must be provided when no levels are available.")
            device = device or self.region_ids.device
            dtype = dtype or torch.float32
            return torch.zeros(*expected_shape, device=device, dtype=dtype)
        return torch.stack(fused, dim=0).mean(dim=0)

    def _expand_time_token(self, Ht: torch.Tensor, T: int, N: int) -> torch.Tensor:
        if Ht.dim() == 2:
            return Ht.unsqueeze(1).expand(-1, T, -1)
        if Ht.dim() == 3:
            if Ht.shape[1] == T and Ht.shape[2] == N:
                return Ht
            raise ValueError(f"Unexpected time token shape {Ht.shape}, expected [B, {T}, {N}]")
        raise ValueError(f"Unexpected time token dims {Ht.dim()}, expected 2 or 3")

    def _expand_space_token(self, Hs: torch.Tensor, T: int, N: int) -> torch.Tensor:
        if Hs.dim() == 2:
            return Hs.unsqueeze(2).expand(-1, T, N)
        if Hs.dim() == 3:
            if Hs.shape[1] == T and Hs.shape[2] == N:
                return Hs
            if Hs.shape[1] == T and Hs.shape[2] == self.num_regions:
                region_ids = self.region_ids.to(device=Hs.device)
                return Hs[:, :, region_ids]
            raise ValueError(f"Unexpected space token shape {Hs.shape}, expected [B, {T}, {N}]")
        raise ValueError(f"Unexpected space token dims {Hs.dim()}, expected 2 or 3")

    def token_gating_single_modal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, T, N, _ = x.shape
        if self._runtime_force_dense or bool(cfg.get("force_dense", False)):
            mask_soft = torch.ones(B, T, N, device=x.device, dtype=x.dtype)
            sparsity_token = mask_soft.new_tensor(0.0)
            return mask_soft, sparsity_token
        Ht = compute_multi_level_time_entropy(
            x, self.time_window_levels, self.entropy_tau, self.entropy_eps, overlap=self.time_window_overlap
        )
        Hs = compute_multi_level_space_entropy(
            x, self.H_p, self.W_p, self.space_window_levels, self.entropy_tau, self.entropy_eps
        )
        Ht_fused = self._fuse_levels(
            Ht["H_time_level"], expected_shape=(B, N), device=x.device, dtype=x.dtype
        )
        Hs_fused = self._fuse_levels(
            Hs["H_space_level"], expected_shape=(B, T), device=x.device, dtype=x.dtype
        )
        Ht_norm = self._normalize(Ht_fused).to(x.dtype)
        Hs_norm = self._normalize(Hs_fused).to(x.dtype)
        H_time_token = self._expand_time_token(Ht_norm, T, N)
        H_space_token = self._expand_space_token(Hs_norm, T, N)
        region_score = H_space_token.mean(dim=1, keepdim=True).expand(-1, T, -1)
        score = (
            cfg.get("alpha_time", 1.0) * H_time_token
            + cfg.get("beta_space", cfg.get("beta_space_coarse", 0.5)) * H_space_token
            + cfg.get("gamma_region", cfg.get("gamma_space_fine", 0.5)) * region_score
        )
        # ---- ablations: token_gating_policy (entropy|uniform|random) ----
        policy = str(cfg.get("token_gating_policy", "entropy")).lower()
        if policy == "random":
            score = torch.rand_like(score)
        elif policy == "uniform":
            idx = torch.arange(score.numel(), device=score.device, dtype=score.dtype).reshape_as(score)
            score = idx / (float(score.numel()) + 1e-12)
        if self.sanitize_nan:
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        mask_soft = []
        rho = float(self._runtime_rho_token) if self._runtime_rho_token is not None else float(cfg.get("rho_token_target", 0.5))
        temperature = float(self._runtime_token_temperature) if self._runtime_token_temperature is not None else float(cfg.get("token_temperature", 0.1))
        for b in range(B):
            flat = score[b].reshape(-1)
            min_keep = int(max(1, float(self.min_keep_ratio) * flat.numel()))
            k = int(rho * flat.numel())
            k = max(min_keep, k)
            k = max(1, min(k, flat.numel()))
            if self.hard_topk:
                topk_idx = torch.topk(flat, k, largest=True).indices
                mask = torch.zeros_like(flat)
                mask[topk_idx] = 1.0
                mask_soft.append(mask.view_as(score[b]))
            else:
                kth = torch.kthvalue(-flat, k).values * -1
                mask_soft.append(torch.sigmoid((score[b] - kth) / temperature))
        mask_soft = torch.stack(mask_soft, dim=0)
        if (not self.hard_topk) and bool(cfg.get("calibrate_keep_mean", True)):
            mask_soft = self._calibrate_keep_mean(mask_soft, target_keep=rho)
        sparsity_token = 1.0 - mask_soft.mean()
        return mask_soft, sparsity_token

    def token_gating_multi_modal(
        self, x: torch.Tensor, modality_slices: Dict[str, Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        modal_stats = {}
        if self._runtime_force_dense or bool(self.cfg.get("force_dense", False)):
            B, T, N, _ = x.shape
            mask_soft = torch.ones(B, T, N, device=x.device, dtype=x.dtype)
            sparsity_token = mask_soft.new_tensor(0.0)
            # still provide per-modality sparsity=0.0
            for name, (s, e) in modality_slices.items():
                modal_stats[name] = {"sparsity": mask_soft.new_tensor(0.0)}
            return mask_soft, sparsity_token, modal_stats
        H_time_token = torch.zeros(x.shape[:3], device=x.device, dtype=x.dtype)
        H_space_token = torch.zeros_like(H_time_token)
        region_ids = self.region_ids.to(x.device)
        w_modal = torch.sigmoid(self.modality_logit)
        modal_id_for_token = torch.zeros(x.shape[2], device=x.device, dtype=torch.long)

        for idx, name in enumerate(self.modalities):
            s, e = modality_slices[name]
            modal_id_for_token[s:e] = idx
            tokens = x[:, :, s:e, :]
            if name == "video":
                Ht = compute_multi_level_time_entropy(
                    tokens, self.time_window_levels, self.entropy_tau, self.entropy_eps, overlap=self.time_window_overlap
                )
                Hs = compute_multi_level_space_entropy(
                    tokens, self.H_p, self.W_p, self.space_window_levels, self.entropy_tau, self.entropy_eps
                )
                Ht_fused = self._fuse_levels(
                    Ht["H_time_level"], expected_shape=(x.shape[0], e - s), device=x.device, dtype=x.dtype
                ).to(x.dtype)
                Hs_fused = self._fuse_levels(
                    Hs["H_space_level"], expected_shape=(x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype
                ).to(x.dtype)
                H_time_token[:, :, s:e] = self._expand_time_token(Ht_fused, x.shape[1], e - s)
                H_space_token[:, :, s:e] = self._expand_space_token(Hs_fused, x.shape[1], e - s)
                modal_stats[name] = {"H_time_mean": Ht_fused.mean()}
            else:
                Ht = compute_multi_level_time_entropy(
                    tokens, self.time_window_levels, self.entropy_tau, self.entropy_eps, overlap=self.time_window_overlap
                )
                Ht_fused = self._fuse_levels(
                    Ht["H_time_level"], expected_shape=(x.shape[0], e - s), device=x.device, dtype=x.dtype
                ).to(x.dtype)
                H_time_token[:, :, s:e] = self._expand_time_token(Ht_fused, x.shape[1], e - s)
                H_space_token[:, :, s:e] = self._expand_time_token(Ht_fused, x.shape[1], e - s)
                modal_stats[name] = {"H_time_mean": Ht_fused.mean()}

        region_score = H_space_token.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        score = (
            self.cfg.get("alpha_time", 1.0) * H_time_token
            + self.cfg.get("beta_space", self.cfg.get("beta_space_coarse", 0.5)) * H_space_token
            + self.cfg.get("gamma_region", self.cfg.get("gamma_space_fine", 0.5)) * region_score
        )
        score = score + self.cfg.get("d_modal", 1.0) * w_modal[modal_id_for_token][None, None, :]
        # ---- ablations: token_gating_policy (entropy|uniform|random) ----
        policy = str(self.cfg.get("token_gating_policy", "entropy")).lower()
        if policy == "random":
            score = torch.rand_like(score)
        elif policy == "uniform":
            idx = torch.arange(score.numel(), device=score.device, dtype=score.dtype).reshape_as(score)
            score = idx / (float(score.numel()) + 1e-12)
        if self.sanitize_nan:
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        mask_soft = []
        rho = float(self._runtime_rho_token) if self._runtime_rho_token is not None else float(self.cfg.get("rho_token_target", 0.5))
        temperature = float(self._runtime_token_temperature) if self._runtime_token_temperature is not None else float(self.cfg.get("token_temperature", 0.1))
        for b in range(x.shape[0]):
            flat = score[b].reshape(-1)
            min_keep = int(max(1, float(self.min_keep_ratio) * flat.numel()))
            k = int(rho * flat.numel())
            k = max(min_keep, k)
            k = max(1, min(k, flat.numel()))
            if self.hard_topk:
                topk_idx = torch.topk(flat, k, largest=True).indices
                mask = torch.zeros_like(flat)
                mask[topk_idx] = 1.0
                mask_soft.append(mask.view_as(score[b]))
            else:
                kth = torch.kthvalue(-flat, k).values * -1
                mask_soft.append(torch.sigmoid((score[b] - kth) / temperature))
        mask_soft = torch.stack(mask_soft, dim=0)
        if (not self.hard_topk) and bool(self.cfg.get("calibrate_keep_mean", True)):
            mask_soft = self._calibrate_keep_mean(mask_soft, target_keep=rho)
        sparsity_token = 1.0 - mask_soft.mean()
        for name, (s, e) in modality_slices.items():
            modal_stats[name]["sparsity"] = 1.0 - mask_soft[:, :, s:e].mean()
        return mask_soft, sparsity_token, modal_stats

    def forward_token_gating(
        self, x: torch.Tensor, modality_slices: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gate = x
        if self.detach_mask:
            x_gate = x_gate.detach()
        if self.gating_fp32:
            x_gate = x_gate.float()
        if self.sanitize_nan:
            x_gate = torch.nan_to_num(x_gate, nan=0.0, posinf=0.0, neginf=0.0)
        if modality_slices:
            mask_soft, sparsity_token, modal_stats = self.token_gating_multi_modal(x_gate, modality_slices)
        else:
            mask_soft, sparsity_token = self.token_gating_single_modal(x_gate)
            modal_stats = {}
        x_gated = x * mask_soft.to(dtype=x.dtype).unsqueeze(-1)
        token_prune = sparsity_token
        token_keep = (1.0 - token_prune)
        return x_gated, {
            "sparsity_token": token_prune,
            "token_prune": token_prune,
            "token_keep": token_keep,
            "mask": mask_soft,
            "modal_stats": modal_stats,
        }

    def compute_L_AST(self, sparsity_token: torch.Tensor, sparsity_head: torch.Tensor, sparsity_ch: torch.Tensor, sparsity_block: torch.Tensor) -> torch.Tensor:
        loss_cfg = self.cfg.get("loss", self.cfg)
        lambda_token = loss_cfg.get("lambda_token", 1.0)
        lambda_head = loss_cfg.get("lambda_head", 0.1)
        lambda_ch = loss_cfg.get("lambda_ch", 0.1)
        lambda_block = loss_cfg.get("lambda_block", 0.1)
        return lambda_token * sparsity_token + lambda_head * sparsity_head + lambda_ch * sparsity_ch + lambda_block * sparsity_block

    def forward(self, token_feat: torch.Tensor, modality_slices: Optional[Dict[str, Tuple[int, int]]] = None) -> ASTOutputs:
        x_gated, stats_token = self.forward_token_gating(token_feat, modality_slices=modality_slices)
        head_weights = torch.sigmoid(self.g_head)
        ch_weights = torch.sigmoid(self.g_ch)
        block_weights = torch.sigmoid(self.g_block)
        sparsity_head = 1.0 - head_weights.mean()
        sparsity_ch = 1.0 - ch_weights.mean()
        sparsity_block = 1.0 - block_weights.mean()
        L_AST = self.compute_L_AST(stats_token["sparsity_token"], sparsity_head, sparsity_ch, sparsity_block)
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
                "modal": stats_token.get("modal_stats", {}),
            },
            L_AST=L_AST,
        )
