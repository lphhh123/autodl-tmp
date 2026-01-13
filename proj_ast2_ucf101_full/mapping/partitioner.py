"""PartitionPlanner with optional fine-grained split + rewrite (SPEC v4 §10)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from mapping.mapping_solver import MappingSolver
from mapping.segments import LayerNode, Segment, build_coarse_segments, build_layer_nodes_from_model, build_segments_from_model
from layout.wafer_layout import WaferLayout
from hw_proxy.layer_hw_proxy import LayerHwProxy


@dataclass
class GraphRewritePlan:
    splits: List[Dict[str, Any]]


class PartitionPlanner:
    def __init__(
        self,
        mapping_solver: MappingSolver,
        wafer_layout: WaferLayout,
        hw_proxy: LayerHwProxy,
        partition_cfg: Any,
    ):
        self.mapping_solver = mapping_solver
        self.wafer_layout = wafer_layout
        self.hw_proxy = hw_proxy
        self.cfg = partition_cfg

    # SPEC v4 §10.2
    def _compute_objective(self, segments: List[Segment], mapping_obj: Dict, cost: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        lat_ms = mapping_obj.get("total_latency_ms", 0.0)
        comm_ms = mapping_obj.get("comm_ms", 0.0)
        per_slot_time = mapping_obj.get("per_slot_time_ms", {})
        w_lat = getattr(self.cfg, "w_latency", 1.0)
        w_comm = getattr(self.cfg, "w_comm", 1e-3)
        w_balance = getattr(self.cfg, "w_balance", 0.0)
        times = torch.tensor(list(per_slot_time.values())) if per_slot_time else torch.tensor([lat_ms])
        imbalance = (times.max() / (times.mean() + 1e-6)).item()
        objective = w_lat * float(lat_ms) + w_comm * float(comm_ms) + w_balance * float(imbalance)
        stats = {
            "lat_ms": float(lat_ms),
            "comm_ms": float(comm_ms),
            "imbalance": float(imbalance),
        }
        return objective, stats

    # SPEC v4 §10.3
    def _build_coarse(self, layer_nodes: List[LayerNode], alpha: torch.Tensor) -> List[Segment]:
        return build_coarse_segments(layer_nodes, alpha, self.cfg)

    # SPEC v4 §10.5
    def _select_split_candidates(self, layer_nodes: List[LayerNode], segments: List[Segment], mapping: List[int], cost: Dict[str, torch.Tensor], eff_specs: Dict[str, torch.Tensor]) -> List[LayerNode]:
        flops_thresh = getattr(self.cfg, "flops_ratio_thresh", 0.3)
        traffic_thresh = getattr(self.cfg, "traffic_ratio_thresh", 0.3)
        candidates = []
        for ln in layer_nodes:
            if not ln.splittable:
                continue
            seg = None
            for s in segments:
                if ln.id in s.layer_ids:
                    seg = s
                    break
            if seg is None or seg.flops <= 0:
                continue
            r_flops = ln.flops / max(1e-9, seg.flops)
            r_traffic = ln.traffic_out_bytes / max(1e-9, seg.traffic_out_bytes)
            if r_flops >= flops_thresh or r_traffic >= traffic_thresh:
                candidates.append(ln)
        return candidates

    def _evaluate(self, segments: List[Segment], eff_specs: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, torch.Tensor], Dict]:
        cost = self.mapping_solver.build_cost_matrix(segments, eff_specs, self.hw_proxy)
        mapping_obj = self.mapping_solver.solve_mapping(
            segments,
            eff_specs,
            self.hw_proxy,
            layout_positions=self.wafer_layout.current_pos_continuous(),
        )
        objective, hw_stats = self._compute_objective(segments, mapping_obj, cost)
        return objective, cost, mapping_obj

    # SPEC v4 §10.6 simulate split
    def _simulate_split_for_layer(self, ln: LayerNode, segments: List[Segment], eff_specs: Dict[str, torch.Tensor]) -> Tuple[bool, float, List[Segment], Dict, Dict[str, Any]]:
        segments_split = []
        split_applied = False
        local_plan = None
        for seg in segments:
            if ln.id not in seg.layer_ids:
                segments_split.append(seg)
                continue
            if seg.layer_ids != [ln.id]:
                segments_split.append(seg)
                continue
            attn_seg = Segment(
                id=seg.id * 10 + 1,
                layer_ids=[ln.id * 2],
                flops=ln.attn_flops,
                bytes=ln.attn_bytes,
                seq_len=seg.seq_len,
                embed_dim=seg.embed_dim,
                num_heads=seg.num_heads,
                mlp_ratio=seg.mlp_ratio,
                precision=seg.precision,
                traffic_in_bytes=ln.attn_bytes,
                traffic_out_bytes=ln.attn_bytes,
                kind="attn",
                block_idx=ln.block_idx,
                keep_factors=ln.keep_factors,
            )
            mlp_seg = Segment(
                id=seg.id * 10 + 2,
                layer_ids=[ln.id * 2 + 1],
                flops=ln.mlp_flops,
                bytes=ln.mlp_bytes,
                seq_len=seg.seq_len,
                embed_dim=seg.embed_dim,
                num_heads=seg.num_heads,
                mlp_ratio=seg.mlp_ratio,
                precision=seg.precision,
                traffic_in_bytes=ln.attn_bytes,
                traffic_out_bytes=ln.mlp_bytes,
                kind="mlp",
                block_idx=ln.block_idx,
                keep_factors=ln.keep_factors,
            )
            segments_split.extend([attn_seg, mlp_seg])
            split_applied = True
            local_plan = {
                "block_idx": ln.block_idx,
                "segments": ["attn", "mlp"],
                "group_to_slot": [],
            }
        if not split_applied:
            return False, 0.0, segments, {}, {}
        obj_base, cost_base, map_base = self._evaluate(segments, eff_specs)
        obj_split, cost_split, map_split = self._evaluate(segments_split, eff_specs)
        gain_ratio = (obj_base - obj_split) / max(1e-6, obj_base)
        local_plan["group_to_slot"] = map_split.get("mapping", [])[:2] if map_split.get("mapping") else []
        return True, gain_ratio, segments_split, map_split, local_plan

    # SPEC v4 §10.7 apply selected
    def _select_accepted_splits(self, split_plans: List[Tuple[int, float, Dict[str, Any], List[Segment], Dict]]) -> List[Tuple[int, float, Dict[str, Any], List[Segment], Dict]]:
        min_gain = getattr(self.cfg, "min_split_gain_ratio", 0.05)
        max_layers = getattr(self.cfg, "max_split_layers", 4)
        filtered = [p for p in split_plans if p[1] >= min_gain]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_layers]

    def _apply_split_plans(self, segments_base: List[Segment], accepted_plans: List[Tuple[int, float, Dict[str, Any], List[Segment], Dict]]) -> Tuple[List[Segment], GraphRewritePlan]:
        if not accepted_plans:
            return segments_base, GraphRewritePlan(splits=[])
        # For simplicity, apply segments from best plan (first one)
        best_plan = accepted_plans[0]
        new_segments = best_plan[3]
        rewrite_plan = GraphRewritePlan(splits=[best_plan[2]])
        return new_segments, rewrite_plan

    def plan(
        self,
        model: torch.nn.Module,
        eff_specs: Dict[str, torch.Tensor],
        alpha: torch.Tensor,
        model_info: Optional[Dict[str, torch.Tensor]] = None,
        use_fine_split: bool = True,
    ) -> Dict[str, Any]:
        layer_nodes = build_layer_nodes_from_model(model, model_info=model_info)
        segments_base = self._build_coarse(layer_nodes, alpha)
        objective_base, cost_base, mapping_base = self._evaluate(segments_base, eff_specs)
        if not use_fine_split:
            return {
                "segments": segments_base,
                "mapping": mapping_base.get("mapping", []),
                "rewrite_plan": None,
                "objective": objective_base,
                "hw_stats": {},
            }

        segments_fine = build_segments_from_model(model, self.cfg, model_info=model_info)
        objective_final, cost_final, mapping_final = self._evaluate(segments_fine, eff_specs)
        rewrite_plan = GraphRewritePlan(
            splits=[{"block_idx": idx, "segments": ["attn", "mlp"]} for idx in range(model.cfg.depth)]
        )
        return {
            "segments": segments_fine,
            "mapping": mapping_final.get("mapping", []),
            "rewrite_plan": rewrite_plan,
            "objective": objective_final,
            "hw_stats": {},
        }
