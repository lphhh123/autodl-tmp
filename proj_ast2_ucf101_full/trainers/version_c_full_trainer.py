"""Full Version-C trainer orchestrating alpha, mapping, and layout."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from datasets.ucf101_dataset import UCF101Dataset, collate_video_batch
from models.ast_losses import ASTRegularizer
from models.vit_video import VideoViT
from utils.distributed_utils import get_device
from utils.logging_utils import setup_logger
from version_c.chip_slots import ChipSlotManager, chip_count_regularizer
from version_c.chip_types import ChipType
from version_c.segmenter import ViTSegmenter
from version_c.mapping_solver import MappingSolver
from version_c.wafer_layout import WaferLayout
from version_c.hw_cost_aggregator import HwCostAggregator
from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.feature_builder import DeviceSpec, LayerConfig
import yaml


def train_version_c_full(cfg):
    device = get_device(cfg.train.device)
    logger = setup_logger(cfg.train.output_dir)

    train_ds = UCF101Dataset(cfg.data.root, cfg.data.split_train, clip_len=cfg.data.clip_len, img_size=cfg.data.img_size, is_train=True)
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=collate_video_batch)

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

    criterion = nn.CrossEntropyLoss()
    ast_reg = ASTRegularizer(
        lambda_token=cfg.ast_loss.lambda_token,
        lambda_head=cfg.ast_loss.lambda_head,
        lambda_channel=cfg.ast_loss.lambda_channel,
        target_sparsity=cfg.ast_loss.target_sparsity,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=True)

    chip_types = []
    for entry in yaml_safe_load(cfg.proxy.gpu_yaml):
        chip_types.append(
            ChipType(
                name=entry["name"],
                peak_flops=entry["peak_flops"],
                peak_bw=entry["peak_bw"],
                mem_size_gb=entry["mem_size_gb"],
                area_mm2=entry["area_mm2"],
                tdp_watt=entry["tdp_watt"],
                energy_per_bit_pj=entry.get("energy_per_bit_pj", 10.0),
            )
        )
    chip_slots = ChipSlotManager(chip_types, num_slots=cfg.version_c.num_slots, temperature=cfg.version_c.temperature).to(device)

    hw_proxy = LayerHwProxy(cfg.proxy.gpu_yaml, cfg.proxy.weight_dir, cfg.proxy.device_name, cfg.proxy.use_power, cfg.proxy.use_mem)
    segmenter = ViTSegmenter(cfg.model)
    mapping_solver = MappingSolver(hw_proxy)

    lambda_cfg = cfg.version_c
    hw_cost = HwCostAggregator(
        hw_proxy,
        lambda_T=lambda_cfg.lambda_T,
        lambda_E=lambda_cfg.lambda_E,
        lambda_mem=lambda_cfg.lambda_mem,
        lambda_area=lambda_cfg.lambda_area,
        lambda_thermal=lambda_cfg.lambda_thermal,
        lambda_comm=lambda_cfg.lambda_comm,
        lambda_chip=lambda_cfg.lambda_chip,
    )

    layout_model = WaferLayout(cfg.layout.wafer_radius_mm, [DeviceSpec(**chip.__dict__) for chip in chip_types]).to(device)
    layout_opt = torch.optim.Adam(layout_model.parameters(), lr=1e-2)

    for outer in range(cfg.train.outer_epochs):
        # A) train theta, s
        for inner in range(cfg.train.inner_epochs):
            for video, target in loader:
                video = video.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with autocast(enabled=True):
                    logits, masks_dict, flops_dict = model(video)
                    loss_task = criterion(logits, target)
                    loss_ast = ast_reg(masks_dict, flops_dict)
                    loss = loss_task + loss_ast
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                break  # keep fast

        alpha = chip_slots(hard=False)
        chip_specs = chip_slots.get_effective_specs(alpha)
        segments = segmenter.build_segments(model)
        mapping, device_times, total_latency, comm_time = mapping_solver.solve_mapping(segments, chip_specs)

        if cfg.version_c.enable_layout:
            for _ in range(5):
                layout_opt.zero_grad()
                layout = layout_model()
                hw_loss, stats = hw_cost.compute_cost(segments, chip_specs, mapping, layout, alpha)
                hw_loss.backward()
                layout_opt.step()
        else:
            layout = layout_model()
            hw_loss, stats = hw_cost.compute_cost(segments, chip_specs, mapping, layout, alpha)

        chip_reg = chip_count_regularizer(alpha, cfg.version_c.lambda_chip)
        total_loss = hw_loss + chip_reg
        logger.info(f"[outer {outer}] hw_loss={hw_loss.item():.4f} chip_reg={chip_reg.item():.4f} mapping={mapping}")


def yaml_safe_load(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("chiplets", [])
