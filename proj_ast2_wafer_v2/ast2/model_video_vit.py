from typing import List, Dict, Any, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

from . import entropy_utils


class PatchEmbed2D(nn.Module):
    """Per-frame 2D patch embedding for video.

    Input:  [B, T, C, H, W]
    Output: tokens [B, T*P, D], along with (T, H_p, W_p)
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.proj(x)  # [B*T, D, H_p, W_p]
        B_T, D, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B*T, P, D]
        P = H_p * W_p
        x = x.view(B, T * P, D)  # [B, T*P, D]
        return x, T, H_p, W_p


class Mlp(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 act_layer=nn.GELU,
                 drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv: [3, B, heads, N, head_dim]
        q, v, k = qkv[0], qkv[2], qkv[1]  # small tweak
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """ViT block with learnable keep_ratio and entropy+Voronoi token pruning."""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)

        # Learnable keep logit (scalar); keep_ratio = sigmoid(keep_logit)
        self.keep_logit = nn.Parameter(torch.tensor(0.0))

        self._last_keep_ratio: float = 1.0

    def get_keep_ratio(self) -> torch.Tensor:
        return torch.sigmoid(self.keep_logit)

    def forward(self,
                x: torch.Tensor,
                num_frames: int,
                height_patches: int,
                width_patches: int) -> torch.Tensor:
        """Forward with token pruning.

        Args:
            x: [B, L, C]
        """
        B, L, C = x.shape
        keep_ratio = torch.sigmoid(self.keep_logit).clamp(0.05, 1.0)
        self._last_keep_ratio = float(keep_ratio.detach().cpu().item())

        # Entropy + Voronoi importance
        importance = entropy_utils.entropy_voronoi_importance(
            self.norm1(x),
            num_frames=num_frames,
            height_patches=height_patches,
            width_patches=width_patches,
        )  # [B, L]

        mask = entropy_utils.topk_mask_from_importance(
            importance, keep_ratio=keep_ratio.item())  # [B, L]

        # Apply mask (hard pruning via zeroing out dropped tokens)
        x_pruned = x * mask.unsqueeze(-1)

        # Standard transformer block on pruned tokens
        x_attn = x_pruned + self.attn(self.norm1(x_pruned))
        x_out = x_attn + self.mlp(self.norm2(x_attn))
        return x_out


class VideoViT(nn.Module):
    """Video Vision Transformer with AST2.0-lite style token pruning.

    This module exposes:
        - get_keep_ratios(): List[Tensor]
        - get_layer_metas(...): List[Dict[str, Any]]
    for integration with hardware proxy.
    """
    def __init__(self,
                 img_size: int = 224,
                 num_frames: int = 8,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 100,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches_per_frame = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches_per_frame * num_frames

        # Simple learned positional embedding (1D over tokens)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Cache for last forward
        self._last_layer_metas: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Introspection used by entropy & hardware modules
    # ------------------------------------------------------------------
    def get_keep_ratios(self) -> List[torch.Tensor]:
        return [b.get_keep_ratio() for b in self.blocks]

    def get_layer_metas(self,
                        input_shape: Tuple[int, int, int, int, int],
                        chip_name: str = "") -> List[Dict[str, Any]]:
        """Return layer_meta list for current architecture.

        input_shape: (B, T, C, H, W)
        chip_name is not used here but is convenient for downstream code.

        This uses cached information from the last forward; if you call this
        before running a forward pass, some fields may be approximate.
        """
        # _last_layer_metas is filled inside forward()
        return list(self._last_layer_metas)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"num_frames mismatch: {T} vs {self.num_frames}"

        tokens, T, H_p, W_p = self.patch_embed(x)  # [B, T*P, D]
        L = tokens.shape[1]
        assert L == self.num_patches

        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        self._last_layer_metas = []
        head_dim = self.embed_dim // self.num_heads
        L_patch = (self.img_size // self.patch_size) ** 2

        for depth_idx, blk in enumerate(self.blocks):
            tokens = blk(tokens, num_frames=T,
                         height_patches=H_p, width_patches=W_p)

            keep_ratio = blk.get_keep_ratio().clamp(0.05, 1.0)
            keep_ratio_val = float(keep_ratio.detach().cpu().item())
            L_eff = int(L_patch * keep_ratio_val)

            layer_meta = {
                "layer_type": "attn",
                "bs": B,
                "img": self.img_size,
                "keep_ratio": keep_ratio_val,
                "L_patch": L_patch,
                "L_eff": L_eff,
                "depth": depth_idx + 1,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "complexity_ratio": 1.0,
                "head_dim": head_dim,
                "tp_world_size": 1,
            }
            self._last_layer_metas.append(layer_meta)

        x = self.norm(tokens)  # [B, L, C]
        # simple global average over tokens
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits
