
from typing import List, Dict, Any, Tuple

import math

from hw_proxy.multi_device_oracle import MultiDeviceHwOracle, estimate_segment_hw_for_device
from .segment_utils import segments_to_edges


class MultiGpuMapper:
    """Simple multi-device mapper + local search for Version-C dummy.

    This class:
      1) receives segment-level meta information;
      2) given a set of device instances (chip ids and types),
         estimates segment cost on each device using the proxy;
      3) performs a naive local search over segment-to-device assignment
         to minimize (makespan + lambda_comm * comm_time).
    """

    def __init__(self, oracle: MultiDeviceHwOracle, lambda_comm: float = 0.01):
        self.oracle = oracle
        self.lambda_comm = lambda_comm

    def initial_mapping(self, num_segments: int, device_ids: List[str]) -> Dict[int, str]:
        # Round-robin initial assignment: seg i -> device_ids[i % len(device_ids)]
        mapping = {}
        for seg_id in range(num_segments):
            mapping[seg_id] = device_ids[seg_id % len(device_ids)]
        return mapping

    def _compute_cost(
        self,
        segments: List[List[Dict[str, Any]]],
        mapping: Dict[int, str],
        device_ids: List[str],
        device_types: Dict[str, str],
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Compute (total_cost, makespan, comm_ms, per_device_ms)."""
        num_segments = len(segments)
        seg_hw = {}  # seg_id -> dict(ms, energy_mj, mem_mb, traffic_bytes)

        per_device_ms = {d: 0.0 for d in device_ids}

        for seg_id, layers in enumerate(segments):
            dev_id = mapping[seg_id]
            dev_type = device_types[dev_id]
            hw = estimate_segment_hw_for_device(self.oracle, layers, dev_type)
            seg_hw[seg_id] = hw
            per_device_ms[dev_id] += hw["ms"]

        makespan = max(per_device_ms.values()) if per_device_ms else 0.0

        # Communication cost
        graph = segments_to_edges(segments)
        edges = graph["edges"]
        comm_ms = 0.0
        for u, v, traffic_bytes in edges:
            dev_u = mapping[u]
            dev_v = mapping[v]
            if dev_u == dev_v:
                continue
            dev_type_u = device_types[dev_u]
            dev_type_v = device_types[dev_v]
            bw_u = self.oracle.device_link_bw(dev_type_u)
            bw_v = self.oracle.device_link_bw(dev_type_v)
            bw = min(bw_u, bw_v)
            comm_ms += (traffic_bytes / (bw + 1e-9)) * 1000.0

        total_cost = makespan + self.lambda_comm * comm_ms
        return total_cost, makespan, comm_ms, per_device_ms

    def local_search(
        self,
        segments: List[List[Dict[str, Any]]],
        device_instances: List[Dict[str, str]],
        max_iters: int = 50,
    ) -> Dict[str, Any]:
        """Run local search mapping for given segments and device instances.

        device_instances: list of dict {id: str, chip_name: str}

        Returns:
            dict with mapping, per_device_ms, makespan, comm_ms, total_cost
        """
        device_ids = [d["id"] for d in device_instances]
        device_types = {d["id"]: d["chip_name"] for d in device_instances}
        num_segments = len(segments)

        if not device_ids or num_segments == 0:
            return {
                "mapping": {},
                "per_device_ms": {},
                "makespan": 0.0,
                "comm_ms": 0.0,
                "total_cost": 0.0,
            }

        mapping = self.initial_mapping(num_segments, device_ids)
        best_mapping = dict(mapping)
        best_total, best_makespan, best_comm, best_per_dev = self._compute_cost(
            segments, mapping, device_ids, device_types
        )

        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            for seg_id in range(num_segments):
                cur_dev = mapping[seg_id]
                for new_dev in device_ids:
                    if new_dev == cur_dev:
                        continue
                    mapping[seg_id] = new_dev
                    total, mk, comm, per_dev = self._compute_cost(
                        segments, mapping, device_ids, device_types
                    )
                    if total + 1e-6 < best_total:
                        best_total, best_makespan, best_comm, best_per_dev = total, mk, comm, per_dev
                        best_mapping = dict(mapping)
                        improved = True
                    else:
                        mapping[seg_id] = cur_dev  # revert

        return {
            "mapping": best_mapping,
            "per_device_ms": best_per_dev,
            "makespan": best_makespan,
            "comm_ms": best_comm,
            "total_cost": best_total,
        }
