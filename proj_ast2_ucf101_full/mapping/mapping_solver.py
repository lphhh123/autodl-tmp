"""Mapping solver (SPEC §9)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from hw_proxy.layer_hw_proxy import LayerHwProxy
from mapping.segments import Segment


class MappingSolver:
    def __init__(self, strategy: str, mem_limit_factor: float):
        self.strategy = strategy
        self.mem_limit_factor = mem_limit_factor

    def build_cost_matrix(self, segments: List[Segment], eff_specs: Dict[str, torch.Tensor], proxy: LayerHwProxy) -> Dict[str, torch.Tensor]:
        layers_cfg = []
        for seg in segments:
            layers_cfg.append(
                {
                    "layer_type": 1,
                    "flops": seg.flops,
                    "bytes": seg.bytes,
                    "embed_dim": seg.embed_dim,
                    "num_heads": seg.num_heads,
                    "mlp_ratio": seg.mlp_ratio,
                    "seq_len": seg.seq_len,
                    "precision": seg.precision,
                }
            )
        cost_np = proxy.predict_layers_batch(layers_cfg)
        lat = torch.tensor(cost_np["lat_ms"], dtype=torch.float32)
        mem = torch.tensor(cost_np["mem_mb"], dtype=torch.float32)
        power = torch.tensor(cost_np["power_w"], dtype=torch.float32)
        # expand to slots
        S = eff_specs["peak_flops"].shape[0]
        lat_ms = lat.unsqueeze(1).repeat(1, S)
        mem_mb = mem.unsqueeze(1).repeat(1, S)
        power_w = power.unsqueeze(1).repeat(1, S)
        return {"lat_ms": lat_ms, "mem_mb": mem_mb, "power_w": power_w}

    def estimate_pipeline_latency(self, mapping: List[int], cost_lat: torch.Tensor, mode: str = "balanced") -> float:
        if mode == "serial":
            total = sum(cost_lat[k, mapping[k]].item() for k in range(len(mapping)))
            return total
        # balanced
        device_time = {}
        for k, d in enumerate(mapping):
            device_time[d] = device_time.get(d, 0.0) + cost_lat[k, d].item()
        return max(device_time.values()) if device_time else 0.0

    def solve_mapping(self, segments: List[Segment], eff_specs: Dict[str, torch.Tensor], proxy: LayerHwProxy, layout_positions: Optional[torch.Tensor] = None, strategy: str = "greedy_local", distance_scale_ms: float = 0.0) -> Dict:
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
        device_time = {s: 0.0 for s in range(S)}
        for k, d in enumerate(mapping):
            device_time[d] += lat_ms[k, d].item()
        comm_ms = 0.0
        if layout_positions is not None:
            for k in range(len(segments) - 1):
                d1, d2 = mapping[k], mapping[k + 1]
                if d1 == d2:
                    continue
                dist = torch.norm(layout_positions[d1] - layout_positions[d2]).item()
                traffic = segments[k].traffic_out_bytes
                eff_bw = min(eff_specs["peak_bw"][d1].item(), eff_specs["peak_bw"][d2].item()) if "peak_bw" in eff_specs else 1.0
                base_time = traffic / (eff_bw + 1e-9) * 1e3
                comm_ms += base_time + dist * distance_scale_ms
        return {"mapping": mapping, "per_slot_time_ms": device_time, "total_latency_ms": current_latency, "comm_ms": comm_ms}

    def build_traffic_matrix(
        self,
        segments: List[Segment],
        mapping: List[int],
        num_slots: int | None = None,
        return_meta: bool = False,
    ) -> torch.Tensor:
        """Aggregate inter-slot traffic for layout export (SPEC v4.3.2 §6.1).

        When ``num_slots`` is provided, a canonical (S,S) matrix aligned to
        slot ids ``0..S-1`` is returned. When omitted, a compact (U,U) matrix
        over the used slots is returned for backward compatibility.
        """

        # Normalize inputs
        mapping_list = [int(x) for x in mapping] if mapping is not None else []
        limit = min(len(segments), len(mapping_list))

        # Handle empty/degenerate cases early
        if limit <= 1:
            if num_slots is not None:
                T = torch.zeros((int(num_slots), int(num_slots)), dtype=torch.float32)
                return (T, {"mode": "canonical", "S": int(num_slots), "mapping_len": len(mapping_list), "sum_bytes": 0.0}) if return_meta else T
            used_slots: List[int] = sorted(set(mapping_list))
            U = len(used_slots)
            T = torch.zeros((U, U), dtype=torch.float32)
            meta = {
                "mode": "compact",
                "U": U,
                "used_slots": used_slots,
                "mapping_len": len(mapping_list),
                "sum_bytes": 0.0,
            }
            return (T, meta) if return_meta else T

        sum_bytes = 0.0

        if num_slots is not None:
            # Branch A: canonical SxS traffic
            S = int(num_slots)
            T = torch.zeros((S, S), dtype=torch.float32)
            for k in range(limit - 1):
                a = int(mapping_list[k])
                b = int(mapping_list[k + 1])
                if a == b:
                    continue
                if not (0 <= a < S) or not (0 <= b < S):
                    raise ValueError(
                        f"[build_traffic_matrix] slot id out of range; mapping[{k}]={a}, mapping[{k+1}]={b}, S={S}"
                    )
                bytes_ab = float(getattr(segments[k], "traffic_out_bytes", 0.0))
                T[a, b] += bytes_ab
                sum_bytes += bytes_ab
            meta = {
                "mode": "canonical",
                "S": S,
                "mapping_len": len(mapping_list),
                "sum_bytes": float(sum_bytes),
            }
            return (T, meta) if return_meta else T

        # Branch B: compact UxU traffic
        used_slots = sorted(set(mapping_list))
        slot_to_idx = {s: i for i, s in enumerate(used_slots)}
        U = len(used_slots)
        T = torch.zeros((U, U), dtype=torch.float32)
        for k in range(limit - 1):
            a = int(mapping_list[k])
            b = int(mapping_list[k + 1])
            if a == b:
                continue
            i = slot_to_idx.get(a)
            j = slot_to_idx.get(b)
            if i is None or j is None:
                continue
            bytes_ab = float(getattr(segments[k], "traffic_out_bytes", 0.0))
            T[i, j] += bytes_ab
            sum_bytes += bytes_ab
        meta = {
            "mode": "compact",
            "U": U,
            "used_slots": used_slots,
            "mapping_len": len(mapping_list),
            "sum_bytes": float(sum_bytes),
        }
        return (T, meta) if return_meta else T

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
