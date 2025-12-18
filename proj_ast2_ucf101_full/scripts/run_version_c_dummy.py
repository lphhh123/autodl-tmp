
"""Version-C dummy script: multi-device mapping + wafer layout.

This script:
  1) builds a VideoViT model and runs one forward pass on random data
     to collect layer meta information;
  2) groups layers into segments;
  3) instantiates a set of chip instances (multi-device system);
  4) runs multi-device mapping with a local-search mapper;
  5) builds a simple device-level communication graph and runs wafer
     layout optimization.

All metrics are based on your proxy models if the weight files exist;
otherwise they use random-initialized proxies (still fine for a dry run).
"""

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


def build_device_instances(chip_types):
    # Example system: two 3090, one 2080ti, one 4090
    instances = [
        {"id": "chip0", "chip_name": "3090"},
        {"id": "chip1", "chip_name": "3090"},
        {"id": "chip2", "chip_name": "2080ti"},
        {"id": "chip3", "chip_name": "4090"},
    ]
    print("[multi-device] instances =", instances)
    return instances


def aggregate_dev_edges(seg_edges, seg_to_dev):
    """Aggregate segment-level edges into device-level edges."""
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



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/version_c_dummy.yaml",
        help="Path to Version-C config yaml",
    )
    args = parser.parse_args()
    cfg_path = args.cfg

    cfg = load_config(cfg_path)
    print(f"==== Version-C dummy: multi-GPU mapping + wafer layout ====")
    print(f"[config] use: {cfg_path}")

    # 1) Layer metas from a dummy model forward
    layer_metas = build_dummy_layer_metas(cfg)
    print(f"[meta] total layers = {len(layer_metas)}")

    # 2) Build segments
    num_segments = cfg.meta.num_dummy_layers if hasattr(cfg.meta, "num_dummy_layers") else 9
    segments = group_layers_into_segments(layer_metas, num_segments=num_segments)
    print(f"[segment] num_segments = {len(segments)}")
    for i, seg in enumerate(segments):
        layer_ids = [lm.layer_id for lm in seg.layers]
        print(f"  - seg_id={i}, layers={layer_ids}")

    # 3) Build multi-device oracle
    md_cfg = cfg.multi_device
    chip_types = md_cfg.chip_types
    instances_cfg = md_cfg.instances

    print(f"[multi-device] chip_types = {chip_types}")
    print(f"[multi-device] instances = {instances_cfg}")

    md_oracle = MultiDeviceOracle(
        gpu_yaml=cfg.proxy.gpu_yaml,
        weight_dir=cfg.proxy.weight_dir,
        chip_types=chip_types,
        instances=instances_cfg,
    )

    # 4) Evaluate an initial mapping (round-robin)
    initial_mapping = build_initial_mapping(segments, instances_cfg)
    print("\n==== Initial mapping (round-robin) ====")
    report_mapping(md_oracle, segments, initial_mapping)

    # 5) Local search mapping (no layout yet)
    best_mapping = improve_mapping_local_search(
        md_oracle,
        segments,
        initial_mapping,
        num_iters=cfg.mapping.get("num_iters", 50),
    )
    print("\n==== After local-search mapping (no layout) ====")
    report_mapping(md_oracle, segments, best_mapping)

    # 6) Build communication graph and initial layout
    edges = build_comm_edges_from_mapping(segments, best_mapping)
    print("\n[layout] edges (u, v, traffic_bytes):")
    for (u, v, traffic) in edges:
        print(f"  {u} -> {v}, traffic={traffic/1e6:.3f} MB")

    layout_cfg = cfg.layout
    wafer_radius = layout_cfg.wafer_radius_mm
    init_positions = init_chip_positions(instances_cfg, wafer_radius=wafer_radius)

    print("\n==== Before layout optimization (with initial positions) ====")
    report_layout(md_oracle, instances_cfg, edges, init_positions)

    # 7) Optimize layout via simple gradient-based heuristics
    opt_positions = optimize_layout(
        md_oracle,
        instances_cfg,
        edges,
        init_positions,
        wafer_radius=wafer_radius,
        num_steps=layout_cfg.get("num_steps", 200),
        lr=layout_cfg.get("lr", 0.05),
    )

    print("\n==== After layout optimization ====")
    report_layout(md_oracle, instances_cfg, edges, opt_positions)

    print("\n==== Version-C dummy pipeline finished (mapping + layout) ====")


if __name__ == "__main__":
    main()
