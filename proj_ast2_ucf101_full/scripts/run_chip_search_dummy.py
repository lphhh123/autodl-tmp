
"""Chip-type & count search dummy script (Version-C high-level search).

This script uses the same building blocks as run_version_c_dummy, but:
  - enumerates small multi-chip configurations from a candidate GPU list;
  - for each configuration, runs mapping + layout;
  - reports the hardware cost so you can see how chip *type & count*
    affect the end-to-end pipeline.

This is still discrete search (not gradient-based), but it matches the
idea that the chiplet configuration is also an optimization variable.
"""

import itertools
import math

import torch

from utils.config import load_config
from ast2.model_video_vit import VideoViT
from hw_proxy.multi_device_oracle import MultiDeviceHwOracle
from mapping.segment_utils import group_layers_into_segments, segments_to_edges
from mapping.mapper_multi_gpu import MultiGpuMapper
from layout.wafer_layout import WaferLayoutOptimizer


def build_dummy_layer_metas(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoViT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.model.num_frames,
        in_chans=cfg.model.in_chans,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    ).to(device)
    model.eval()
    with torch.no_grad():
        video = torch.randn(
            2,
            cfg.model.in_chans,
            cfg.model.num_frames,
            cfg.model.img_size,
            cfg.model.img_size,
            device=device,
        )
        _ = model(video)
    metas = model.get_layer_metas()
    return metas


def aggregate_dev_edges(seg_edges, seg_to_dev):
    dev_edge_map = {}
    for u, v, traffic in seg_edges:
        du = seg_to_dev[u]
        dv = seg_to_dev[v]
        key = tuple(sorted((du, dv)))
        dev_edge_map[key] = dev_edge_map.get(key, 0.0) + traffic

    dev_edges = []
    dev_id_to_idx = {}
    for idx, dev_id in enumerate(sorted(set([d for pair in dev_edge_map.keys() for d in pair]))):
        dev_id_to_idx[dev_id] = idx

    for (du, dv), traffic in dev_edge_map.items():
        i = dev_id_to_idx[du]
        j = dev_id_to_idx[dv]
        dev_edges.append((i, j, traffic))
    return dev_edges, dev_id_to_idx


def build_device_instances_for_combo(combo):
    instances = []
    for idx, chip_name in enumerate(combo):
        instances.append({"id": f"chip{idx}", "chip_name": chip_name})
    return instances


def evaluate_combo(cfg, layer_metas, oracle, combo):
    # 1) Segments
    num_segments = cfg.meta.num_dummy_layers if hasattr(cfg.meta, "num_dummy_layers") else 9
    segments = group_layers_into_segments(layer_metas, num_segments=num_segments)

    # 2) Mapper
    mapper = MultiGpuMapper(oracle, lambda_comm=float(cfg.mapping.lambda_comm))
    device_instances = build_device_instances_for_combo(combo)
    result = mapper.local_search(
        segments, device_instances, max_iters=cfg.mapping.max_ls_iters
    )
    mapping = result["mapping"]
    per_dev_ms = result["per_device_ms"]
    makespan = result["makespan"]
    comm_ms = result["comm_ms"]
    total_cost = result["total_cost"]

    # 3) Layout
    seg_graph = segments_to_edges(segments)
    seg_edges = seg_graph["edges"]
    dev_edges, dev_id_to_idx = aggregate_dev_edges(seg_edges, mapping)

    # Device area for layout
    device_area_mm2 = {}
    for inst in device_instances:
        dev_id = inst["id"]
        chip_name = inst["chip_name"]
        area = oracle.device_area(chip_name)
        device_area_mm2[dev_id] = area

    # Map dev_edges indices back to instance indices
    devid_to_inst_idx = {inst["id"]: idx for idx, inst in enumerate(device_instances)}
    dev_edges_idx = []
    id_list = list(dev_id_to_idx.keys())
    for i, j, traffic in dev_edges:
        dev_id_i = id_list[i]
        dev_id_j = id_list[j]
        inst_i = devid_to_inst_idx[dev_id_i]
        inst_j = devid_to_inst_idx[dev_id_j]
        dev_edges_idx.append((inst_i, inst_j, traffic))

    layout_opt = WaferLayoutOptimizer(
        wafer_radius_mm=float(cfg.layout.wafer_radius_mm),
        lr=0.1,
        steps=50,
        lambda_comm=1e-6,
    )
    layout_res = layout_opt.optimize(
        device_instances=device_instances,
        dev_edges=dev_edges_idx,
        device_area_mm2=device_area_mm2,
    )

    return {
        "combo": combo,
        "mapping": mapping,
        "per_dev_ms": per_dev_ms,
        "makespan": makespan,
        "comm_ms": comm_ms,
        "total_cost": total_cost,
        "positions": layout_res["positions"],
    }


def enumerate_combos(candidate_types, max_chips):
    combos = []
    for k in range(1, max_chips + 1):
        for tup in itertools.product(candidate_types, repeat=k):
            combos.append(list(tup))
    return combos


def main():
    cfg = load_config("configs/version_c_chip_search_dummy.yaml")
    print("==== Version-C chip search dummy ====")

    # 1) Layer metas from dummy model
    layer_metas = build_dummy_layer_metas(cfg)
    print(f"[meta] total layers = {len(layer_metas)}")

    # 2) Oracle (one per candidate type internally)
    candidate_types = cfg.hw.candidate_types
    oracle = MultiDeviceHwOracle(gpu_yaml=cfg.hw.gpu_yaml, chip_types=candidate_types)

    # 3) Enumerate small chip combos
    max_chips = cfg.hw.max_chips
    combos = enumerate_combos(candidate_types, max_chips)
    print(f"[search] evaluate {len(combos)} chip configurations (up to {max_chips} chips)")

    results = []
    for combo in combos:
        res = evaluate_combo(cfg, layer_metas, oracle, combo)
        results.append(res)
        print(
            f"  combo={combo}: total_cost={res['total_cost']:.3f} ms, "
            f"makespan={res['makespan']:.3f} ms, comm_ms={res['comm_ms']:.3f} ms"
        )

    # Sort by total_cost
    results_sorted = sorted(results, key=lambda r: r["total_cost"])
    print("\n==== Top-5 chip configurations by total_cost ====")
    for i, res in enumerate(results_sorted[:5]):
        combo = res["combo"]
        print(
            f"[{i}] combo={combo}, total_cost={res['total_cost']:.3f} ms, "
            f"makespan={res['makespan']:.3f} ms, comm_ms={res['comm_ms']:.3f} ms"
        )

    print("\n==== Version-C chip search dummy finished ====")


if __name__ == "__main__":
    main()
