# profile_layerwise_power.py
import argparse
import csv
import time
import subprocess
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import amp

PATCH_SIZE = 16
IN_CHANS = 3


# ----------------- 功耗采样 ----------------- #

def get_gpu_power_w(device_index: int = 0):
    """
    调用 nvidia-smi 获取某块 GPU 当前功耗（W）。
    失败则返回 None，不会中断主流程。
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8")
        return float(out.strip())
    except Exception as e:
        print(f"[warn] get_gpu_power_w failed: {e}")
        return None


# ----------------- 层级算子定义 ----------------- #

class PatchEmbedLayer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)                         # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)         # [B, L, embed_dim]
        return x


class SelfAttnLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # 使用 PyTorch 自带 MultiheadAttention，batch_first=True
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # x: [B, L, C]
        out, _ = self.attn(x, x, x)
        return out


class MlpLayer(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        # x: [B, L, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ----------------- 数据结构 & 读取 ----------------- #

@dataclass
class LayerSample:
    device: str
    cfg: str
    layer_type: str   # "patch_embed" / "attn" / "mlp"
    prec: str         # "fp16" / "fp32"
    img: int
    bs: int
    keep_ratio: float
    L_patch: int
    L_eff: int
    status: str
    depth: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    complexity_ratio: float
    head_dim: int
    tp_world_size: int


def load_layer_dataset(path: str):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                LayerSample(
                    device=row["device"],
                    cfg=row["cfg"],
                    layer_type=row["layer_type"],
                    prec=row["prec"],
                    img=int(row["img"]),
                    bs=int(row["bs"]),
                    keep_ratio=float(row["keep_ratio"]),
                    L_patch=int(row["L_patch"]),
                    L_eff=int(row["L_eff"]),
                    status=row.get("status", "ok"),
                    depth=int(row["depth"]),
                    embed_dim=int(row["embed_dim"]),
                    num_heads=int(row["num_heads"]),
                    mlp_ratio=float(row["mlp_ratio"]),
                    complexity_ratio=float(row["complexity_ratio"]),
                    head_dim=int(row["head_dim"]),
                    tp_world_size=int(row["tp_world_size"]),
                )
            )
    return rows


# ----------------- 单层 profile：CUDA event + 功耗 ----------------- #

def profile_one_layer(sample: LayerSample,
                      runs: int = 20,
                      warmup: int = 8,
                      device_index: int = 0):
    """
    对单条 layer 配置做多次前向：
      - 用 CUDA events 统计平均前向时间（ms_event）【仅用于能量计算】
      - 用 nvidia-smi 统计平均功耗（avg_power_w）
      - 计算每次前向能量（mJ）
    返回 (status, ms_event, avg_power_w, energy_mj)
    """
    # 只处理 status == ok 的样本
    if sample.status != "ok":
        return "skip", None, None, None

    # 设置精度
    prec = sample.prec.lower()
    if prec == "fp16":
        dtype = torch.float16
        use_amp = True
    else:
        dtype = torch.float32
        use_amp = False

    device = torch.device("cuda", device_index)

    # 构建输入与模块
    layer_type = sample.layer_type
    img = sample.img
    bs = sample.bs
    L_eff = sample.L_eff
    C = sample.embed_dim
    heads = max(1, sample.num_heads)
    mlp_ratio = sample.mlp_ratio

    try:
        if layer_type == "patch_embed":
            # 输入: [B, 3, H, W]
            x = torch.randn(bs, IN_CHANS, img, img, device=device, dtype=dtype)
            layer = PatchEmbedLayer(
                img_size=img,
                patch_size=PATCH_SIZE,
                in_chans=IN_CHANS,
                embed_dim=C,
            ).to(device=device, dtype=dtype)
        elif layer_type == "attn":
            # 输入: [B, L_eff, C]
            x = torch.randn(bs, L_eff, C, device=device, dtype=dtype)
            layer = SelfAttnLayer(
                embed_dim=C,
                num_heads=heads,
            ).to(device=device, dtype=dtype)
        elif layer_type == "mlp":
            # 输入: [B, L_eff, C]
            x = torch.randn(bs, L_eff, C, device=device, dtype=dtype)
            layer = MlpLayer(
                embed_dim=C,
                mlp_ratio=mlp_ratio,
            ).to(device=device, dtype=dtype)
        else:
            print(f"[warn] unknown layer_type {layer_type}, skip")
            return "skip", None, None, None

        layer.eval()
    except RuntimeError as e:
        # 构建中就 OOM / 其他问题
        print(f"[oom] build layer failed for {sample.cfg} {layer_type}: {e}")
        return "oom", None, None, None

    # Amp 上下文
    if use_amp:
        ctx = amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, *exc): return False
        def _ctx_gen():
            return DummyCtx()
        ctx = _ctx_gen()

    # CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # 预热：不计入统计
    with torch.no_grad():
        with ctx:
            for _ in range(warmup):
                _ = layer(x)
        torch.cuda.synchronize(device)

    times = []
    powers = []

    with torch.no_grad():
        for _ in range(runs):
            # 采样功耗（在计时区间外，不影响 events）
            p_before = get_gpu_power_w(device_index)

            # 确保上一轮结束
            torch.cuda.synchronize(device)
            start_evt.record()

            with ctx:
                _ = layer(x)

            end_evt.record()
            torch.cuda.synchronize(device)

            # GPU 前向时间（ms）
            elapsed_ms = start_evt.elapsed_time(end_evt)
            times.append(elapsed_ms)

            p_after = get_gpu_power_w(device_index)
            if p_before is not None and p_after is not None:
                powers.append(0.5 * (p_before + p_after))

    if not times:
        return "fail", None, None, None

    ms_event = sum(times) / len(times)
    if powers:
        avg_power_w = sum(powers) / len(powers)
        # 能量: P(W) * t(s) = J，转成 mJ
        energy_j = avg_power_w * (ms_event / 1000.0)
        energy_mj = energy_j * 1000.0
    else:
        avg_power_w = None
        energy_mj = None

    return "ok", ms_event, avg_power_w, energy_mj


# ----------------- 主流程 ----------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default="RTX2080Ti",
        help="逻辑 device 名字，只写入 CSV，不影响实际 GPU 选择",
    )
    ap.add_argument(
        "--data_csv",
        default="layer_dataset.csv",
        help="已有的层级数据 CSV（只用结构/配置信息）",
    )
    ap.add_argument(
        "--out_csv",
        default="layer_power_dataset.csv",
        help="输出的功耗数据 CSV",
    )
    ap.add_argument("--runs", type=int, default=20, help="每个样本实际计时/计功耗的轮数")
    ap.add_argument("--warmup", type=int, default=8, help="预热轮数，不计入统计")
    ap.add_argument("--gpu_index", type=int, default=0, help="使用哪块物理 GPU（通常 0）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    samples = load_layer_dataset(args.data_csv)
    print(f"[data] load {len(samples)} layer samples from {args.data_csv}")

    # 只保留 status=ok
    samples_ok = [s for s in samples if s.status == "ok"]
    print(f"[data] status=ok samples = {len(samples_ok)}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "device",
            "cfg",
            "layer_type",
            "prec",
            "img",
            "bs",
            "keep_ratio",
            "L_patch",
            "L_eff",
            "avg_power_w",
            "energy_mj",
            "ms_event",
            "status",
            "depth",
            "embed_dim",
            "num_heads",
            "mlp_ratio",
            "complexity_ratio",
            "head_dim",
            "tp_world_size",
            "runs",
            "warmup",
        ])

        total = len(samples_ok)
        for i, s in enumerate(samples_ok):
            print(
                f"\n=== [{i+1}/{total}] "
                f"{s.cfg} {s.layer_type} {s.prec} "
                f"img={s.img} bs={s.bs} keep={s.keep_ratio:.3f} ===",
                flush=True,
            )

            status, ms_event, avg_power_w, energy_mj = profile_one_layer(
                s,
                runs=args.runs,
                warmup=args.warmup,
                device_index=args.gpu_index,
            )

            print(
                f"[{args.device} {s.layer_type} {s.prec}] "
                f"img={s.img} bs={s.bs} keep={s.keep_ratio:.3f} "
                f"→ status={status}, ms={ms_event}, P={avg_power_w}, E(mJ)={energy_mj}",
                flush=True,
            )

            w.writerow([
                args.device,
                s.cfg,
                s.layer_type,
                s.prec,
                s.img,
                s.bs,
                f"{s.keep_ratio:.6f}",
                s.L_patch,
                s.L_eff,
                "" if avg_power_w is None else f"{avg_power_w:.6f}",
                "" if energy_mj is None else f"{energy_mj:.6f}",
                "" if ms_event is None else f"{ms_event:.6f}",
                status,
                s.depth,
                s.embed_dim,
                s.num_heads,
                f"{s.mlp_ratio:.6f}",
                f"{s.complexity_ratio:.6f}",
                s.head_dim,
                s.tp_world_size,
                args.runs,
                args.warmup,
            ])

    print(f"\n[done] power dataset written to {args.out_csv}")


if __name__ == "__main__":
    main()
