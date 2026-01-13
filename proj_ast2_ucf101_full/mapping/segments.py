"""Segment definitions and builders (SPEC §8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from models.video_vit import VideoViT


@dataclass
class LayerNode:
    id: int
    layer_type: str
    flops: float
    bytes: float
    seq_len: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    precision: int
    traffic_in_bytes: float
    traffic_out_bytes: float
    splittable: bool = False
    block_idx: Optional[int] = None
    attn_flops: float = 0.0
    mlp_flops: float = 0.0
    attn_bytes: float = 0.0
    mlp_bytes: float = 0.0
    keep_factors: Optional[Dict[str, float]] = None


@dataclass
class Segment:
    id: int
    layer_ids: List[int]
    flops: float
    bytes: float
    seq_len: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    precision: int
    traffic_in_bytes: float
    traffic_out_bytes: float
    kind: str = "other"
    block_idx: Optional[int] = None
    keep_factors: Optional[Dict[str, float]] = None
    can_split_fine: bool = False
    fine_groups: Optional[int] = None


@dataclass
class SegmentEdge:
    src: int
    dst: int
    traffic_bytes: float


@dataclass
class SegmentGraph:
    segments: List[Segment]
    edges: List[SegmentEdge]


def _extract_keep_factors(model_info: Optional[Dict[str, torch.Tensor]], depth: int) -> Dict[str, List[float]]:
    if not model_info:
        return {
            "token_keep": [1.0] * depth,
            "head_keep": [1.0] * depth,
            "ch_keep": [1.0] * depth,
            "block_keep": [1.0] * depth,
        }

    token_mask = model_info.get("token_mask")
    token_keep = float(token_mask.float().mean().item()) if token_mask is not None else 1.0

    head_weights = model_info.get("head_weights")
    ch_weights = model_info.get("ch_weights")
    block_weights = model_info.get("block_weights")

    head_keep = (
        [float(w.mean().item()) for w in head_weights]
        if head_weights is not None
        else [1.0] * depth
    )
    ch_keep = (
        [float(w.mean().item()) for w in ch_weights]
        if ch_weights is not None
        else [1.0] * depth
    )
    block_keep = (
        [float(w.item()) for w in block_weights]
        if block_weights is not None
        else [1.0] * depth
    )

    if len(head_keep) != depth:
        head_keep = [head_keep[0]] * depth
    if len(ch_keep) != depth:
        ch_keep = [ch_keep[0]] * depth
    if len(block_keep) != depth:
        block_keep = [block_keep[0]] * depth

    return {
        "token_keep": [token_keep] * depth,
        "head_keep": head_keep,
        "ch_keep": ch_keep,
        "block_keep": block_keep,
    }


def build_segments_from_model(model: VideoViT, cfg, model_info: Optional[Dict[str, torch.Tensor]] = None, precision: int = 1) -> List[Segment]:
    layer_nodes = build_layer_nodes_from_model(model, model_info=model_info, precision=precision)
    segments: List[Segment] = []
    seg_id = 0
    for node in layer_nodes:
        keep = node.keep_factors or {}
        attn_seg = Segment(
            id=seg_id,
            layer_ids=[node.id * 2],
            flops=node.attn_flops,
            bytes=node.attn_bytes,
            seq_len=node.seq_len,
            embed_dim=node.embed_dim,
            num_heads=node.num_heads,
            mlp_ratio=node.mlp_ratio,
            precision=node.precision,
            traffic_in_bytes=node.attn_bytes,
            traffic_out_bytes=node.attn_bytes,
            kind="attn",
            block_idx=node.block_idx,
            keep_factors=keep,
        )
        seg_id += 1
        mlp_seg = Segment(
            id=seg_id,
            layer_ids=[node.id * 2 + 1],
            flops=node.mlp_flops,
            bytes=node.mlp_bytes,
            seq_len=node.seq_len,
            embed_dim=node.embed_dim,
            num_heads=node.num_heads,
            mlp_ratio=node.mlp_ratio,
            precision=node.precision,
            traffic_in_bytes=node.attn_bytes,
            traffic_out_bytes=node.mlp_bytes,
            kind="mlp",
            block_idx=node.block_idx,
            keep_factors=keep,
        )
        seg_id += 1
        segments.extend([attn_seg, mlp_seg])
    return segments


def build_layer_nodes_from_model(model: VideoViT, model_info: Optional[Dict[str, torch.Tensor]] = None, precision: int = 1) -> List[LayerNode]:
    depth = model.cfg.depth
    seq_len = model.num_tokens
    embed_dim = model.cfg.embed_dim
    num_heads = model.cfg.num_heads
    mlp_ratio = model.cfg.mlp_ratio
    bytes_per_elem = 2 if precision == 1 else 4
    keep = _extract_keep_factors(model_info, depth)
    nodes: List[LayerNode] = []
    for idx in range(depth):
        token_keep = keep["token_keep"][idx]
        head_keep = keep["head_keep"][idx]
        ch_keep = keep["ch_keep"][idx]
        block_keep = keep["block_keep"][idx]

        attn_base_flops = 4.0 * seq_len * (embed_dim**2)
        mlp_base_flops = 2.0 * seq_len * embed_dim * (embed_dim * mlp_ratio)
        attn_flops = attn_base_flops * (token_keep**2) * head_keep * block_keep
        mlp_flops = mlp_base_flops * token_keep * ch_keep * block_keep

        attn_bytes = seq_len * embed_dim * bytes_per_elem * token_keep
        mlp_bytes = seq_len * embed_dim * bytes_per_elem * token_keep

        nodes.append(
            LayerNode(
                id=idx,
                layer_type="block",
                flops=attn_flops + mlp_flops,
                bytes=attn_bytes + mlp_bytes,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                precision=precision,
                traffic_in_bytes=attn_bytes,
                traffic_out_bytes=mlp_bytes,
                splittable=True,
                block_idx=idx,
                attn_flops=attn_flops,
                mlp_flops=mlp_flops,
                attn_bytes=attn_bytes,
                mlp_bytes=mlp_bytes,
                keep_factors={
                    "token_keep": token_keep,
                    "head_keep": head_keep,
                    "ch_keep": ch_keep,
                    "block_keep": block_keep,
                },
            )
        )
    return nodes


def build_coarse_segments(layer_nodes: List[LayerNode], alpha: torch.Tensor, partition_cfg: Any = None) -> List[Segment]:
    empty_idx = alpha.shape[1] - 1
    p_active = 1.0 - alpha[:, empty_idx]
    active_threshold = float(getattr(partition_cfg, "active_threshold", 0.5)) if partition_cfg else 0.5
    active_mask = p_active > active_threshold
    if not bool(active_mask.any()):
        active_mask[torch.argmax(p_active)] = True
    num_active = int(active_mask.sum().item())

    F_total = sum(ln.flops for ln in layer_nodes)
    F_target = F_total / max(1, num_active)

    segments: List[Segment] = []
    seg_id = 0
    acc_flops = 0.0
    acc_bytes = 0.0
    acc_layers: List[int] = []
    for ln in layer_nodes:
        acc_flops += ln.flops
        acc_bytes += ln.bytes
        acc_layers.append(ln.id)
        if acc_flops >= F_target:
            segments.append(
                Segment(
                    id=seg_id,
                    layer_ids=acc_layers.copy(),
                    flops=acc_flops,
                    bytes=acc_bytes,
                    seq_len=ln.seq_len,
                    embed_dim=ln.embed_dim,
                    num_heads=ln.num_heads,
                    mlp_ratio=ln.mlp_ratio,
                    precision=ln.precision,
                    traffic_in_bytes=ln.traffic_in_bytes,
                    traffic_out_bytes=ln.traffic_out_bytes,
                    kind="other",
                    block_idx=None,
                    keep_factors=None,
                )
            )
            seg_id += 1
            acc_flops = 0.0
            acc_bytes = 0.0
            acc_layers = []
    if acc_layers:
        ln = layer_nodes[-1]
        segments.append(
            Segment(
                id=seg_id,
                layer_ids=acc_layers.copy(),
                flops=acc_flops,
                bytes=acc_bytes,
                seq_len=ln.seq_len,
                embed_dim=ln.embed_dim,
                num_heads=ln.num_heads,
                mlp_ratio=ln.mlp_ratio,
                precision=ln.precision,
                traffic_in_bytes=ln.traffic_in_bytes,
                traffic_out_bytes=ln.traffic_out_bytes,
                kind="other",
                block_idx=None,
                keep_factors=None,
            )
        )
    return segments
