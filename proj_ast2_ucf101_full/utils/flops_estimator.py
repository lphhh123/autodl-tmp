"""Lightweight FLOPs/Bytes estimators for VideoViT components (SPEC_version_c_full §7)."""
from __future__ import annotations

from typing import Tuple


def estimate_attention_flops_bytes(seq_len: int, embed_dim: int, num_heads: int, precision_bytes: int = 2) -> Tuple[float, float]:
    """Approximate attention FLOPs/bytes for a ViT block."""
    # QKV projection + attention + output projection
    flops_qkv = 3.0 * seq_len * embed_dim * embed_dim
    flops_attn = 2.0 * num_heads * (seq_len ** 2) * (embed_dim // num_heads)
    flops_out = seq_len * embed_dim * embed_dim
    flops = flops_qkv + flops_attn + flops_out
    bytes_ = seq_len * embed_dim * precision_bytes  # activation volume
    return flops, bytes_


def estimate_mlp_flops_bytes(seq_len: int, embed_dim: int, mlp_ratio: float, precision_bytes: int = 2) -> Tuple[float, float]:
    """Approximate MLP FLOPs/bytes for a ViT block."""
    hidden = int(embed_dim * mlp_ratio)
    flops_fc1 = seq_len * embed_dim * hidden
    flops_act = seq_len * hidden  # GELU approx
    flops_fc2 = seq_len * hidden * embed_dim
    flops = flops_fc1 + flops_act + flops_fc2
    bytes_ = seq_len * embed_dim * precision_bytes
    return flops, bytes_


def estimate_block_flops_bytes(seq_len: int, embed_dim: int, num_heads: int, mlp_ratio: float, precision_bytes: int = 2) -> Tuple[float, float]:
    attn_f, attn_b = estimate_attention_flops_bytes(seq_len, embed_dim, num_heads, precision_bytes)
    mlp_f, mlp_b = estimate_mlp_flops_bytes(seq_len, embed_dim, mlp_ratio, precision_bytes)
    return attn_f + mlp_f, attn_b + mlp_b
