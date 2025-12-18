"""Phase 3 mapping baseline trainer."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.ucf101_dataset import UCF101Dataset, collate_video_batch
from models.vit_video import VideoViT
from utils.distributed_utils import get_device
from utils.logging_utils import setup_logger
from version_c.segmenter import ViTSegmenter
from version_c.mapping_solver import MappingSolver
from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.feature_builder import LayerConfig


def run_phase3(cfg):
    device = get_device(cfg.train.device)
    logger = setup_logger(cfg.train.output_dir)
    ds = UCF101Dataset(cfg.data.root, cfg.data.split_train, clip_len=cfg.data.clip_len, img_size=cfg.data.img_size, is_train=True)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=collate_video_batch)

    model = VideoViT(
        img_size=cfg.model.img_size,
        num_frames=cfg.model.num_frames,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        use_ast2=cfg.model.use_ast2,
        ast=getattr(cfg.model, "ast", {}),
    ).to(device)

    segmenter = ViTSegmenter(cfg.model, strategy="uniform_block_group")
    segments = segmenter.build_segments(model)

    hw_proxy = LayerHwProxy(cfg.proxy.gpu_yaml, cfg.proxy.weight_dir, cfg.proxy.device_name, cfg.proxy.use_power, cfg.proxy.use_mem)
    solver = MappingSolver(hw_proxy)

    devices = list(hw_proxy.device_map.values()) if hasattr(hw_proxy, "device_map") else []
    mapping, device_times, total_latency, comm_time = solver.solve_mapping(segments, devices)
    logger.info(f"Mapping result: {mapping}")
    logger.info(f"Device times: {device_times}, total latency: {total_latency}, comm: {comm_time}")
