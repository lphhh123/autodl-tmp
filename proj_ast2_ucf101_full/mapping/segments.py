"""Segment definitions and builders (SPEC §8)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from models.vit_video import VideoViT


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


@dataclass
class SegmentEdge:
    src: int
    dst: int
    traffic_bytes: float


@dataclass
class SegmentGraph:
    segments: List[Segment]
    edges: List[SegmentEdge]


def _layer_flops_bytes(seq_len: int, embed_dim: int, num_heads: int, mlp_ratio: float) -> tuple[float, float]:
    # approximate per SPEC §8
    attn_flops = 4.0 * seq_len * (embed_dim ** 2)
    mlp_flops = 2.0 * seq_len * embed_dim * (embed_dim * mlp_ratio)
    flops = attn_flops + mlp_flops
    bytes_ = seq_len * embed_dim * 4.0  # simple activation estimate
    return flops, bytes_


def build_segments_from_model(model: VideoViT, cfg) -> List[Segment]:
    # back-compat coarse grouping
    layer_nodes = build_layer_nodes_from_model(model)
    return build_coarse_segments(layer_nodes, {}, getattr(cfg, "partition", None))


# SPEC v4: build LayerNodes and coarse segments


def build_layer_nodes_from_model(model: VideoViT, precision: int = 1) -> List[LayerNode]:
    depth = model.cfg.depth
    seq_len = model.num_tokens
    embed_dim = model.cfg.embed_dim
    num_heads = model.cfg.num_heads
    mlp_ratio = model.cfg.mlp_ratio
    bytes_per_elem = 2 if precision == 1 else 4
    nodes: List[LayerNode] = []
    node_id = 0
    for _ in range(depth):
        attn_flops = 4.0 * seq_len * (embed_dim ** 2)
        attn_bytes = seq_len * embed_dim * bytes_per_elem
        nodes.append(
            LayerNode(
                id=node_id,
                layer_type="attn",
                flops=attn_flops,
                bytes=attn_bytes,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                precision=precision,
                traffic_in_bytes=attn_bytes,
                traffic_out_bytes=attn_bytes,
                splittable=True,
            )
        )
        node_id += 1
        mlp_flops = 2.0 * seq_len * embed_dim * (embed_dim * mlp_ratio)
        mlp_bytes = seq_len * embed_dim * bytes_per_elem
        nodes.append(
            LayerNode(
                id=node_id,
                layer_type="mlp",
                flops=mlp_flops,
                bytes=mlp_bytes,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                precision=precision,
                traffic_in_bytes=mlp_bytes,
                traffic_out_bytes=mlp_bytes,
                splittable=True,
            )
        )
        node_id += 1
    return nodes


def build_coarse_segments(layer_nodes: List[LayerNode], eff_specs: Dict[str, torch.Tensor], partition_cfg: Any = None) -> List[Segment]:
    F_total = sum(ln.flops for ln in layer_nodes)
    chip_used_prob = None
    if eff_specs and isinstance(eff_specs, dict) and "mem_gb" in eff_specs:
        chip_used_prob = torch.ones_like(eff_specs["mem_gb"])
    num_active = max(1, int(chip_used_prob.sum().item())) if chip_used_prob is not None else 1
    F_target = F_total / num_active
    segments: List[Segment] = []
    seg_id = 0
    acc_flops = 0.0
    acc_bytes = 0.0
    acc_layers: List[int] = []
    bytes_per_elem = 2
    seq_len = layer_nodes[0].seq_len if layer_nodes else 0
    embed_dim = layer_nodes[0].embed_dim if layer_nodes else 0
    num_heads = layer_nodes[0].num_heads if layer_nodes else 0
    mlp_ratio = layer_nodes[0].mlp_ratio if layer_nodes else 0.0
    precision = layer_nodes[0].precision if layer_nodes else 1
    min_layers_per_segment = getattr(partition_cfg, "min_layers_per_segment", 1) if partition_cfg else 1
    for ln in layer_nodes:
        acc_flops += ln.flops
        acc_bytes += ln.bytes
        acc_layers.append(ln.id)
        if acc_flops >= F_target and len(acc_layers) >= min_layers_per_segment:
            traffic = seq_len * embed_dim * bytes_per_elem
            segments.append(
                Segment(
                    id=seg_id,
                    layer_ids=acc_layers.copy(),
                    flops=acc_flops,
                    bytes=acc_bytes,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    precision=precision,
                    traffic_in_bytes=traffic,
                    traffic_out_bytes=traffic,
                )
            )
            seg_id += 1
            acc_flops = 0.0
            acc_bytes = 0.0
            acc_layers = []
    if acc_layers:
        traffic = seq_len * embed_dim * bytes_per_elem
        segments.append(
            Segment(
                id=seg_id,
                layer_ids=acc_layers.copy(),
                flops=acc_flops,
                bytes=acc_bytes,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                precision=precision,
                traffic_in_bytes=traffic,
                traffic_out_bytes=traffic,
            )
        )
    return segments
