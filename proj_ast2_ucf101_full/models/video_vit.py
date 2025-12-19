"""VideoViT backbone with AST2.0-lite pruning (SPEC_version_c_full)."""
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
            self.ast_pruner = ASTPruner(ast_cfg, cfg.embed_dim, cfg.num_heads, cfg.depth, num_patches)
            self.ast_pruner.cfg.setdefault("patch_grid_h", grid[0])
            self.ast_pruner.cfg.setdefault("patch_grid_w", grid[1])
        else:
            self.ast_pruner = None

    def _embed_video(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        patches = F.unfold(x.view(b * t, c, h, w), kernel_size=self.cfg.patch_size, stride=self.cfg.patch_size)
        patches = patches.transpose(1, 2)  # [B*T, N, patch_dim]
        tokens = self.patch_embed(patches)  # [B*T, N, C]
        tokens = tokens.view(b, t, self.num_tokens, -1)
        return tokens

    def _forward_tokens(self, tokens: torch.Tensor, ast_out: Optional[ASTOutputs], b: int, t: int):
        tokens = tokens.view(b * t, self.num_tokens, -1)
        cls_tokens = self.cls_token.expand(b * t, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1)
        seq = seq + self.pos_embed[:, : seq.size(1), :]
        seq = self.pos_drop(seq)

        if ast_out is None:
            head_w = [torch.ones(self.cfg.num_heads, device=tokens.device) for _ in range(len(self.blocks))]
            ch_w = [torch.ones(int(self.cfg.embed_dim * self.cfg.mlp_ratio), device=tokens.device) for _ in range(len(self.blocks))]
            block_w = [torch.tensor(1.0, device=tokens.device) for _ in range(len(self.blocks))]
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
        logits = self.head(cls_out)  # [B*T, num_classes]
        logits = logits.view(b, t, -1).mean(dim=1)  # aggregate over frames
        return logits, head_w, ch_w, block_w, sparsity

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        b, t, c, h, w = x.shape
        tokens = self._embed_video(x)  # [B, T, N, C]
        ast_out: Optional[ASTOutputs] = None
        L_AST = torch.tensor(0.0, device=x.device)
        if self.use_ast and self.ast_pruner is not None:
            ast_out = self.ast_pruner(tokens)
            token_mask = ast_out.token_mask.unsqueeze(-1)  # [B, T, N, 1]
            tokens = tokens * token_mask
        logits, head_w, ch_w, block_w, sparsity = self._forward_tokens(tokens, ast_out, b, t)

        if not return_intermediate:
            return logits
        L_ast_val = ast_out.L_AST if ast_out is not None else L_AST
        info = {
            "L_AST": L_ast_val,
            "token_feat": tokens,
            "gates": {
                "token_mask": ast_out.token_mask if ast_out is not None else torch.ones(b, t, self.num_tokens, device=x.device),
                "head_weights": head_w,
                "ch_weights": ch_w,
                "block_weights": block_w,
                "sparsity": sparsity,
            },
            "ast_stats": {
                "L_AST": L_ast_val,
                "sparsity_token": sparsity.get("token") if sparsity else None,
                "sparsity_head": sparsity.get("head") if sparsity else None,
                "sparsity_ch": sparsity.get("ch") if sparsity else None,
                "sparsity_block": sparsity.get("block") if sparsity else None,
            },
        }
        return logits, info


class VideoAudioAST(nn.Module):
    def __init__(
        self,
        img_size: int,
        num_frames: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        patch_size: int,
        audio_feat_dim: int,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        use_ast_prune: bool = True,
        ast_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.cfg = VideoViTConfig(
            img_size=img_size,
            num_frames=num_frames,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            in_chans=in_chans,
            drop_rate=drop_rate,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            use_ast_prune=use_ast_prune,
            ast_cfg=ast_cfg or {},
        )
        patch_dim = in_chans * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        num_patches_video = (img_size // patch_size) ** 2
        num_patches_audio = 1
        num_patches_total = num_patches_video + num_patches_audio
        self.num_tokens = num_patches_total
        self.num_tokens_video = num_patches_video
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_total + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, attn_drop=attn_drop, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.audio_proj = nn.Linear(audio_feat_dim, embed_dim)

        self.use_ast = use_ast_prune
        self.ast_pruner: Optional[ASTPruner]
        if self.use_ast:
            ast_cfg = dict(ast_cfg or {})
            ast_cfg["num_modalities"] = 2
            ast_cfg["modalities"] = ["video", "audio"]
            ast_cfg["num_patches_video"] = num_patches_video
            ast_cfg["num_patches_audio"] = num_patches_audio
            ast_cfg.setdefault("patch_grid_h", img_size // patch_size)
            ast_cfg.setdefault("patch_grid_w", img_size // patch_size)
            self.ast_pruner = ASTPruner(ast_cfg, embed_dim, num_heads, depth, num_patches_total)
        else:
            self.ast_pruner = None

    def _embed_video(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        patches = F.unfold(x.view(b * t, c, h, w), kernel_size=self.cfg.patch_size, stride=self.cfg.patch_size)
        patches = patches.transpose(1, 2)  # [B*T, N, patch_dim]
        tokens = self.patch_embed(patches)  # [B*T, N, C]
        tokens = tokens.view(b, t, self.num_tokens_video, -1)
        return tokens

    def forward(self, x_video: torch.Tensor, x_audio: torch.Tensor, return_intermediate: bool = False):
        b, t, c, h, w = x_video.shape
        v_tokens = self._embed_video(x_video)
        a_tokens = self.audio_proj(x_audio).unsqueeze(2)  # [B, T, 1, C]
        tokens = torch.cat([v_tokens, a_tokens], dim=2)

        ast_out: Optional[ASTOutputs] = None
        L_AST = torch.tensor(0.0, device=x_video.device)
        modality_slices = {"video": (0, self.num_tokens_video), "audio": (self.num_tokens_video, self.num_tokens_video + 1)}
        if self.use_ast and self.ast_pruner is not None:
            ast_out = self.ast_pruner(tokens, modality_slices=modality_slices)
            token_mask = ast_out.token_mask.unsqueeze(-1)  # [B, T, N, 1]
            tokens = tokens * token_mask

        tokens = tokens.view(b * t, self.num_tokens, -1)
        cls_tokens = self.cls_token.expand(b * t, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1)
        seq = seq + self.pos_embed[:, : seq.size(1), :]
        seq = self.pos_drop(seq)

        if ast_out is None:
            head_w = [torch.ones(self.cfg.num_heads, device=x_video.device) for _ in range(len(self.blocks))]
            ch_w = [torch.ones(int(self.cfg.embed_dim * self.cfg.mlp_ratio), device=x_video.device) for _ in range(len(self.blocks))]
            block_w = [torch.tensor(1.0, device=x_video.device) for _ in range(len(self.blocks))]
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
        logits = logits.view(b, t, -1).mean(dim=1)

        if not return_intermediate:
            return logits
        L_ast_val = ast_out.L_AST if ast_out is not None else L_AST
        info = {
            "L_AST": L_ast_val,
            "token_feat": tokens.view(b, t, self.num_tokens, -1),
            "modality_slices": modality_slices,
            "gates": {
                "token_mask": ast_out.token_mask if ast_out is not None else torch.ones(b, t, self.num_tokens, device=x_video.device),
                "head_weights": head_w,
                "ch_weights": ch_w,
                "block_weights": block_w,
                "sparsity": sparsity,
            },
            "ast_stats": {
                "L_AST": L_ast_val,
                "sparsity_token": sparsity.get("token") if sparsity else None,
                "sparsity_head": sparsity.get("head") if sparsity else None,
                "sparsity_ch": sparsity.get("ch") if sparsity else None,
                "sparsity_block": sparsity.get("block") if sparsity else None,
            },
        }
        return logits, info
