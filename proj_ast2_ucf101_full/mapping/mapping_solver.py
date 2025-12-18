"""Mapping solver with optional fine split + channel rewire (SPEC_version_c_full §8)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from hw_proxy.layer_hw_proxy import LayerHwProxy
from mapping.segments import Segment


class MappingSolver:
    def __init__(self, strategy: str, mem_limit_factor: float, fine_split_margin: float = 0.05):
        self.strategy = strategy
        self.mem_limit_factor = mem_limit_factor
        self.fine_split_margin = fine_split_margin

    def build_cost_matrix(self, segments: List[Segment], eff_specs: Dict[str, torch.Tensor], proxy: LayerHwProxy) -> Dict[str, torch.Tensor]:
        """Build per-segment, per-slot latency/mem/power predictions."""
        S = eff_specs["peak_flops"].shape[0]
        lat_ms_list = []
        mem_mb_list = []
        power_w_list = []
        for seg in segments:
            lat_row = []
            mem_row = []
            power_row = []
            for s in range(S):
                seg_cfg = {
                    "layer_type": 1,
                    "flops": seg.flops,
                    "bytes": seg.bytes,
                    "embed_dim": seg.embed_dim,
                    "num_heads": seg.num_heads,
                    "mlp_ratio": seg.mlp_ratio,
                    "seq_len": seg.seq_len,
                    "precision": seg.precision,
                }
                dev_specs = {k: eff_specs[k][s].item() for k in eff_specs}
                pred = proxy.predict_segment(seg_cfg, dev_specs)
                lat_row.append(pred["lat_ms"])
                mem_row.append(pred["mem_mb"])
                power_row.append(pred["power_w"])
            lat_ms_list.append(lat_row)
            mem_mb_list.append(mem_row)
            power_w_list.append(power_row)
        return {
            "lat_ms": torch.tensor(lat_ms_list, dtype=torch.float32),
            "mem_mb": torch.tensor(mem_mb_list, dtype=torch.float32),
            "power_w": torch.tensor(power_w_list, dtype=torch.float32),
        }

    def estimate_pipeline_latency(self, mapping: List[int], cost_lat: torch.Tensor, mode: str = "balanced") -> float:
        if mode == "serial":
            total = sum(cost_lat[k, mapping[k]].item() for k in range(len(mapping)))
            return total
        # balanced
        device_time = {}
        for k, d in enumerate(mapping):
            device_time[d] = device_time.get(d, 0.0) + cost_lat[k, d].item()
        return max(device_time.values()) if device_time else 0.0

    def solve_mapping(self, segments: List[Segment], eff_specs: Dict[str, torch.Tensor], proxy: LayerHwProxy, layout_positions: Optional[torch.Tensor] = None, strategy: str = "greedy_local", distance_scale_ms: float = 0.0, cfg: Optional[Any] = None) -> Dict:
        cfg = cfg or {}
        cost = self.build_cost_matrix(segments, eff_specs, proxy)
        lat_ms = cost["lat_ms"]
        mem_mb = cost["mem_mb"]
        K, S = lat_ms.shape
        mapping = [k % S for k in range(K)]
        current_latency = self.estimate_pipeline_latency(mapping, lat_ms)
        improved = True
        while improved:
            improved = False
            for k in range(K):
                curr_d = mapping[k]
                best_d = curr_d
                best_latency = current_latency
                for d in range(S):
                    if d == curr_d:
                        continue
                    if self._violates_mem(mapping, k, d, mem_mb, eff_specs):
                        continue
                    old_d = mapping[k]
                    mapping[k] = d
                    new_latency = self.estimate_pipeline_latency(mapping, lat_ms)
                    if new_latency + 1e-6 < best_latency:
                        best_latency = new_latency
                        best_d = d
                    mapping[k] = old_d
                if best_d != curr_d:
                    mapping[k] = best_d
                    current_latency = best_latency
                    improved = True
        segments_final = segments
        rewire_meta = []
        if cfg.get("enable_fine_split", True):
            fine_groups = cfg.get("fine_groups", 2)
            updated_segments: List[Segment] = []
            updated_mapping: List[int] = []
            for idx, seg in enumerate(segments):
                if seg.can_split_fine:
                    use_fine, fine_segments, meta = self.try_fine_split_segment(seg, eff_specs, proxy, layout_positions, fine_groups, distance_scale_ms)
                    if use_fine:
                        rewire_meta.append(meta)
                        updated_segments.extend(fine_segments)
                        updated_mapping.extend(meta["group_to_slot"])
                        continue
                updated_segments.append(seg)
                updated_mapping.append(mapping[idx])
            if len(updated_segments) != len(segments):
                segments_final = updated_segments
                cost = self.build_cost_matrix(segments_final, eff_specs, proxy)
                lat_ms = cost["lat_ms"]
                mem_mb = cost["mem_mb"]
                mapping = updated_mapping
                current_latency = self.estimate_pipeline_latency(mapping, lat_ms)
        device_time = {s: 0.0 for s in range(S)}
        for k, d in enumerate(mapping):
            device_time[d] += lat_ms[k, d].item()
        comm_ms = self._estimate_comm(mapping, segments_final, eff_specs, layout_positions, distance_scale_ms)
        return {"mapping": mapping, "per_slot_time_ms": device_time, "total_latency_ms": current_latency, "comm_ms": comm_ms, "segments": segments_final, "rewire_meta": rewire_meta}

    def _violates_mem(self, mapping: List[int], k_idx: int, new_d: int, mem_mb: torch.Tensor, eff_specs: Dict[str, torch.Tensor]) -> bool:
        tmp_map = mapping.copy()
        tmp_map[k_idx] = new_d
        S = eff_specs["mem_gb"].shape[0]
        usage = [0.0 for _ in range(S)]
        for k, d in enumerate(tmp_map):
            usage[d] = max(usage[d], mem_mb[k, d].item())
        for s in range(S):
            limit = eff_specs["mem_gb"][s].item() * 1024 * self.mem_limit_factor
            if usage[s] > limit:
                return True
        return False

    def _estimate_comm(self, mapping: List[int], segments: List[Segment], eff_specs: Dict[str, torch.Tensor], layout_positions: Optional[torch.Tensor], distance_scale_ms: float) -> float:
        comm_ms = 0.0
        if layout_positions is None or "peak_bw" not in eff_specs:
            return comm_ms
        for k in range(len(segments) - 1):
            d1, d2 = mapping[k], mapping[k + 1]
            if d1 == d2:
                continue
            dist = torch.norm(layout_positions[d1] - layout_positions[d2]).item()
            traffic = segments[k].traffic_out_bytes
            eff_bw = min(eff_specs["peak_bw"][d1].item(), eff_specs["peak_bw"][d2].item())
            comm_ms += traffic / (eff_bw + 1e-9) * 1e3 + dist * distance_scale_ms
        return comm_ms

    def try_fine_split_segment(self, segment: Segment, eff_specs: Dict[str, torch.Tensor], proxy: LayerHwProxy, layout_positions: Optional[torch.Tensor], G: int, distance_scale_ms: float) -> tuple[bool, List[Segment], Dict[str, Any]]:
        coarse_cfg = {
            "layer_type": 1,
            "flops": segment.flops,
            "bytes": segment.bytes,
            "embed_dim": segment.embed_dim,
            "num_heads": segment.num_heads,
            "mlp_ratio": segment.mlp_ratio,
            "seq_len": segment.seq_len,
            "precision": segment.precision,
        }
        coarse_cost = []
        for s in range(eff_specs["peak_flops"].shape[0]):
            coarse_cost.append(proxy.predict_segment(coarse_cfg, {k: eff_specs[k][s].item() for k in eff_specs})["lat_ms"])
        coarse_best = min(coarse_cost)
        fine_segments: List[Segment] = []
        for g in range(G):
            fine_segments.append(
                Segment(
                    id=segment.id * 10 + g,
                    layer_ids=segment.layer_ids,
                    flops=segment.flops / G,
                    bytes=segment.bytes / G,
                    seq_len=segment.seq_len,
                    embed_dim=segment.embed_dim,
                    num_heads=segment.num_heads,
                    mlp_ratio=segment.mlp_ratio,
                    precision=segment.precision,
                    traffic_in_bytes=segment.traffic_in_bytes / G,
                    traffic_out_bytes=segment.traffic_out_bytes / G,
                    can_split_fine=False,
                    fine_groups=None,
                )
            )
        group_to_slot: List[int] = []
        fine_lat = []
        for g_seg in fine_segments:
            best_slot = 0
            best_lat = float("inf")
            for s in range(eff_specs["peak_flops"].shape[0]):
                pred = proxy.predict_segment(
                    {
                        "layer_type": 1,
                        "flops": g_seg.flops,
                        "bytes": g_seg.bytes,
                        "embed_dim": g_seg.embed_dim,
                        "num_heads": g_seg.num_heads,
                        "mlp_ratio": g_seg.mlp_ratio,
                        "seq_len": g_seg.seq_len,
                        "precision": g_seg.precision,
                    },
                    {k: eff_specs[k][s].item() for k in eff_specs},
                )
                if pred["lat_ms"] < best_lat:
                    best_lat = pred["lat_ms"]
                    best_slot = s
            group_to_slot.append(best_slot)
            fine_lat.append(best_lat)
        fine_latency = max(fine_lat)
        fine_comm = 0.0
        if layout_positions is not None and len(group_to_slot) >= 2:
            for i in range(len(group_to_slot) - 1):
                d1, d2 = group_to_slot[i], group_to_slot[i + 1]
                if d1 == d2:
                    continue
                dist = torch.norm(layout_positions[d1] - layout_positions[d2]).item()
                eff_bw = min(eff_specs["peak_bw"][d1].item(), eff_specs["peak_bw"][d2].item()) if "peak_bw" in eff_specs else 1.0
                fine_comm += (segment.traffic_out_bytes / G) / (eff_bw + 1e-9) * 1e3 + dist * distance_scale_ms
        cost_fine = fine_latency + fine_comm
        margin = self.fine_split_margin * coarse_best
        use_fine = (cost_fine + margin) < coarse_best
        meta = {
            "segment_id": segment.id,
            "num_groups": G,
            "perm": list(range(segment.embed_dim)),
            "group_to_slot": group_to_slot,
        }
        return use_fine, (fine_segments if use_fine else [segment]), meta
