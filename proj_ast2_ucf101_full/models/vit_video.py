"""VideoViT backbone with AST2.0-lite pruning (SPEC_version_c.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast2_pruner import ASTOutputs, ASTPruner


@dataclass
class VideoViTConfig:
    img_size: int = 224
    num_frames: int = 8
    num_classes: int = 101
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_size: int = 16
    in_chans: int = 3
    drop_rate: float = 0.0
    attn_drop: float = 0.0
    drop_path_rate: float = 0.0
    use_ast_prune: bool = False
    ast_cfg: Optional[Dict] = None


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, ch_weight: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = h * ch_weight.unsqueeze(0).unsqueeze(0)
        h = self.act(h)
        h = self.fc2(h)
        return h


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, head_weight: torch.Tensor, ch_weight: torch.Tensor, block_weight: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        head_scale = head_weight.mean()
        h = h * head_scale
        x = x + block_weight * h
        mlp_out = self.mlp(self.norm2(x), ch_weight)
        x = x + block_weight * mlp_out
        return x


class VideoViT(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        cfg = VideoViTConfig(**kwargs)
        self.cfg = cfg
        patch_dim = cfg.in_chans * cfg.patch_size * cfg.patch_size
        self.patch_embed = nn.Linear(patch_dim, cfg.embed_dim)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.num_tokens = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, attn_drop=cfg.attn_drop, drop=cfg.drop_rate)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self.use_ast = cfg.use_ast_prune
        self.ast_pruner: Optional[ASTPruner]
        if self.use_ast:
            grid = (cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size)
            ast_cfg = cfg.ast_cfg or {}
            self.ast_pruner = ASTPruner(ast_cfg, cfg.embed_dim, cfg.num_heads, cfg.depth, grid[0], grid[1])
        else:
            self.ast_pruner = None

    def _embed_video(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        patches = F.unfold(x.view(b * t, c, h, w), kernel_size=self.cfg.patch_size, stride=self.cfg.patch_size)
        patches = patches.transpose(1, 2)  # [B*T, N, patch_dim]
        tokens = self.patch_embed(patches)  # [B*T, N, C]
        tokens = tokens.view(b, t, self.num_tokens, -1)
        return tokens

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        b, t, c, h, w = x.shape
        tokens = self._embed_video(x)  # [B, T, N, C]
        ast_out: Optional[ASTOutputs] = None
        L_AST = torch.tensor(0.0, device=x.device)
        if self.use_ast and self.ast_pruner is not None:
            ast_out = self.ast_pruner(tokens)
            token_mask = ast_out.token_mask.unsqueeze(-1)  # [B, T, N, 1]
            tokens = tokens * token_mask
        tokens = tokens.view(b * t, self.num_tokens, -1)
        cls_tokens = self.cls_token.expand(b * t, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1)
        seq = seq + self.pos_embed[:, : seq.size(1), :]
        seq = self.pos_drop(seq)

        if ast_out is None:
            head_w = [torch.ones(self.cfg.num_heads, device=x.device) for _ in range(len(self.blocks))]
            ch_w = [torch.ones(int(self.cfg.embed_dim * self.cfg.mlp_ratio), device=x.device) for _ in range(len(self.blocks))]
            block_w = [torch.tensor(1.0, device=x.device) for _ in range(len(self.blocks))]
            sparsity = {}
        else:
            head_w = ast_out.head_weights
            ch_w = ast_out.ch_weights
            block_w = ast_out.block_weights
            sparsity = ast_out.sparsity

        for idx, blk in enumerate(self.blocks):
            seq = blk(seq, head_w[idx], ch_w[idx], block_w[idx])
        seq = self.norm(seq)
        cls_out = seq[:, 0]
        logits = self.head(cls_out)

        if not return_intermediate:
            return logits
        info = {
            "token_feat": tokens.view(b, t, self.num_tokens, -1),
            "gates": {
                "token_mask": ast_out.token_mask if ast_out is not None else torch.ones(b, t, self.num_tokens, device=x.device),
                "head_weights": head_w,
                "ch_weights": ch_w,
                "block_weights": block_w,
                "sparsity": sparsity,
            },
            "ast_stats": {
                "L_AST": ast_out.L_AST if ast_out is not None else L_AST,
                "sparsity_token": sparsity.get("token") if sparsity else None,
                "sparsity_head": sparsity.get("head") if sparsity else None,
                "sparsity_ch": sparsity.get("ch") if sparsity else None,
                "sparsity_block": sparsity.get("block") if sparsity else None,
            },
        }
        return logits, info
