from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import math

from hw_proxy.layer_proxy import LayerHwProxy


@dataclass
class Chiplet:
    name: str
    mem_capacity: float  # e.g. MB
    compute_capacity: float  # relative (e.g. peak_flops)
    link_bw: float  # GB/s
    idx: int = 0


@dataclass
class Segment:
    idx: int
    layer_indices: List[int]  # indices of layers in model
    flops: float
    params: float
    # predicted cost per chiplet (compute-only, no comm)
    cost_per_chip: Dict[int, float]  # chip_idx -> latency (ms)
    mem_per_chip: Dict[int, float]   # chip_idx -> peak_mem (MB)


def build_segments(layer_metas: List[Dict[str, Any]],
                   layers_per_segment: int = 2) -> List[Segment]:
    """Group consecutive layers into segments.

    This is a simple implementation; you can replace it with more
    sophisticated graph-based segmentation if needed.
    """
    segments: List[Segment] = []
    cur = 0
    idx = 0
    while cur < len(layer_metas):
        l_inds = list(range(cur, min(cur + layers_per_segment, len(layer_metas))))
        # Use FLOPs proxy: embed_dim^2 * num_heads * L_eff for attention approx
        flops = 0.0
        params = 0.0
        for li in l_inds:
            meta = layer_metas[li]
            d = float(meta["embed_dim"])
            h = float(meta["num_heads"])
            L_eff = float(meta["L_eff"])
            flops += 4.0 * d * d * h * L_eff  # rough
        segments.append(Segment(
            idx=idx,
            layer_indices=l_inds,
            flops=flops,
            params=params,
            cost_per_chip={},
            mem_per_chip={},
        ))
        idx += 1
        cur += layers_per_segment
    return segments


def estimate_segment_costs(
    segments: List[Segment],
    chips: List[Chiplet],
    layer_metas: List[Dict[str, Any]],
    proxy: LayerHwProxy,
):
    """Fill cost_per_chip/mem_per_chip for each segment using proxy."""
    for seg in segments:
        for chip in chips:
            total_ms = 0.0
            peak_mem = 0.0
            for li in seg.layer_indices:
                meta = layer_metas[li]
                pred = proxy.predict_layer(chip.name, meta)
                total_ms += pred["ms"]
                peak_mem = max(peak_mem, pred["mem"])
            seg.cost_per_chip[chip.idx] = total_ms
            seg.mem_per_chip[chip.idx] = peak_mem


def greedy_initial_mapping(
    segments: List[Segment],
    chips: List[Chiplet],
) -> Dict[int, int]:
    """Greedy mapping: assign segments to chiplets by compute load and mem fit.

    Returns:
        mapping: seg_idx -> chip_idx
    """
    # Track per-chip assigned latency and mem
    load_ms = {chip.idx: 0.0 for chip in chips}
    load_mem = {chip.idx: 0.0 for chip in chips}

    # Sort segments by descending flops
    segs_sorted = sorted(segments, key=lambda s: s.flops, reverse=True)

    mapping: Dict[int, int] = {}

    for seg in segs_sorted:
        best_chip = None
        best_score = math.inf
        for chip in chips:
            cost = seg.cost_per_chip[chip.idx]
            mem = seg.mem_per_chip[chip.idx]
            # simple mem constraint: assume chip.mem_capacity
            if load_mem[chip.idx] + mem > chip.mem_capacity:
                continue
            # score: resulting total latency on that chip
            score = load_ms[chip.idx] + cost
            if score < best_score:
                best_score = score
                best_chip = chip.idx
        if best_chip is None:
            # if no chip can fit, assign to the least loaded one and ignore mem
            best_chip = min(load_ms, key=load_ms.get)
        mapping[seg.idx] = best_chip
        load_ms[best_chip] += seg.cost_per_chip[best_chip]
        load_mem[best_chip] = max(load_mem[best_chip], seg.mem_per_chip[best_chip])

    return mapping


def mapping_cost(
    segments: List[Segment],
    chips: List[Chiplet],
    mapping: Dict[int, int],
    inter_seg_bytes: Dict[Tuple[int, int], float],
    link_bw: float,
    lambda_comm: float = 1.0,
) -> float:
    """Compute total cost (compute + comm) for a given mapping.

    Args:
        inter_seg_bytes: (seg_u, seg_v) -> bytes to communicate between them
        link_bw: GB/s (assumed same for all links for simplicity)
    """
    # compute cost per chip
    compute_ms = {chip.idx: 0.0 for chip in chips}
    for seg in segments:
        chip_idx = mapping[seg.idx]
        compute_ms[chip_idx] += seg.cost_per_chip[chip_idx]

    total_compute = max(compute_ms.values())  # critical path

    # comm cost: sum of bytes / bw for edges crossing chips
    comm_ms = 0.0
    for (u, v), bytes_ in inter_seg_bytes.items():
        cu = mapping[u]
        cv = mapping[v]
        if cu == cv:
            continue
        t = (bytes_ / (link_bw * 1e9)) * 1e3  # seconds -> ms
        comm_ms += t

    return total_compute + lambda_comm * comm_ms


def local_search_refine(
    segments: List[Segment],
    chips: List[Chiplet],
    mapping: Dict[int, int],
    inter_seg_bytes: Dict[Tuple[int, int], float],
    link_bw: float,
    lambda_comm: float = 1.0,
    max_iters: int = 50,
) -> Dict[int, int]:
    """Simple hill-climbing local search over segment-to-chip mapping."""
    best_mapping = dict(mapping)
    best_cost = mapping_cost(
        segments, chips, best_mapping,
        inter_seg_bytes, link_bw, lambda_comm
    )

    for _ in range(max_iters):
        improved = False
        for seg in segments:
            cur_chip = best_mapping[seg.idx]
            for chip in chips:
                if chip.idx == cur_chip:
                    continue
                candidate = dict(best_mapping)
                candidate[seg.idx] = chip.idx
                cost = mapping_cost(
                    segments, chips, candidate,
                    inter_seg_bytes, link_bw, lambda_comm
                )
                if cost < best_cost:
                    best_cost = cost
                    best_mapping = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_mapping
