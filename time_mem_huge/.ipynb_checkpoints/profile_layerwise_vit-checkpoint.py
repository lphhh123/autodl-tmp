# profile_layerwise_vit.py
import argparse
import csv
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import amp
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

"""
作用：
  基于 pruned_cfg_grid.csv（结构化剪枝后的整网配置），
  在一组 (keep_ratio, img, bs, prec) 上，
  对 3 类“层”：patch_embed / attention / mlp 做独立 profile，记录：
    - 层的前向中位数时延（ms）
    - 层的峰值显存（MB）
    - OOM 情况（status="oom"）
  输出 layer_dataset.csv，作为后续 per-layer MLP 代理的训练数据。

注意：
  - 每层独立构建，以 timm 的 PatchEmbed / Attention / Mlp 为基元。
  - 输入张量按当前剪枝配置构造（embed_dim / num_heads / mlp_ratio / keep_ratio / img / bs）。
  - 所有前向都在 no_grad() 下执行；fp16 使用 autocast。
"""

TOKEN_KEEP_LIST = [1.0, 0.75, 0.5, 0.375]
PATCH_SIZE = 16
IN_CHANS = 3


def measure_module(mod, input_shape, prec="fp16", runs=30, warmup=10):
    """
    在给定 input_shape 上多次跑 mod(x)，返回：
      - 中位数时延（ms）
      - 峰值显存（MB）
    """
    device = torch.device("cuda")
    mod = mod.to(device).eval()

    if prec == "fp16":
        dtype = torch.float16
        ctx = amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        dtype = torch.float32
        ctx = nullcontext()

    x = torch.randn(*input_shape, device=device, dtype=dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    with torch.no_grad(), ctx:
        # 预热
        for _ in range(warmup):
            _ = mod(x)
            torch.cuda.synchronize()
        # 正式计时
        for _ in range(runs):
            t0 = time.time()
            _ = mod(x)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)

    ms = float(np.median(times))
    peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return ms, peak_mem_mb


def load_cfgs(path):
    cfgs = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            embed_dim = int(row["embed_dim"])
            num_heads = int(row["num_heads"])
            head_dim = row.get("head_dim")
            if head_dim is None or head_dim == "":
                head_dim = embed_dim // max(num_heads, 1)
            else:
                head_dim = int(head_dim)

            tp_world_size = row.get("tp_world_size")
            if tp_world_size is None or tp_world_size == "":
                tp_world_size = 1
            else:
                tp_world_size = int(tp_world_size)

            cfgs.append(
                {
                    "cfg": row["cfg"],
                    "depth": int(row["depth"]),
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "mlp_ratio": float(row["mlp_ratio"]),
                    "complexity_ratio": float(row["complexity_ratio"]),
                    "head_dim": head_dim,
                    "tp_world_size": tp_world_size,
                }
            )
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="RTX2080Ti")
    ap.add_argument("--cfg_csv", default="pruned_cfg_grid.csv")
    ap.add_argument("--out_csv", default="layer_dataset.csv")
    ap.add_argument("--imgs", default="160,176,192,208,224")
    ap.add_argument("--bss", default="1,8,16,24,32,48,64")
    ap.add_argument("--fp16", action="store_true", help="仅测 fp16（调试用）")
    ap.add_argument("--fp32", action="store_true", help="仅测 fp32（调试用）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    imgs = [int(x) for x in args.imgs.split(",") if x]
    bss = [int(x) for x in args.bss.split(",") if x]

    if args.fp16 and not args.fp32:
        precs = ["fp16"]
    elif args.fp32 and not args.fp16:
        precs = ["fp32"]
    else:
        precs = ["fp16", "fp32"]

    cfgs = load_cfgs(args.cfg_csv)
    print(f"[info] total cfgs from {args.cfg_csv}: {len(cfgs)}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "device",
                "cfg",
                "layer_type",
                "prec",
                "img",
                "bs",
                "keep_ratio",
                "L_patch",
                "L_eff",
                "ms",
                "peak_mem_mb",
                "status",
                "depth",
                "embed_dim",
                "num_heads",
                "mlp_ratio",
                "complexity_ratio",
                "head_dim",
                "tp_world_size",
            ]
        )

        for ci, cfg in enumerate(cfgs):
            print(f"\n=== cfg {ci+1}/{len(cfgs)}: {cfg['cfg']} ===", flush=True)
            depth = cfg["depth"]
            embed_dim = cfg["embed_dim"]
            num_heads = cfg["num_heads"]
            mlp_ratio = cfg["mlp_ratio"]

            for prec in precs:
                for keep_ratio in TOKEN_KEEP_LIST:
                    for img in imgs:
                        # token 数
                        assert img % PATCH_SIZE == 0, "img must be divisible by patch size"
                        num_patches = (img // PATCH_SIZE) ** 2
                        L_patch = num_patches       # 不含 cls
                        if keep_ratio >= 0.999:
                            L_eff = L_patch + 1
                        else:
                            L_keep = max(1, int(L_patch * keep_ratio))
                            L_eff = L_keep + 1      # + cls

                        # 构建三类模块（一次构建，多次复用）
                        patch_mod = PatchEmbed(
                            img_size=img,
                            patch_size=PATCH_SIZE,
                            in_chans=IN_CHANS,
                            embed_dim=embed_dim,
                        )

                        attn_mod = Attention(
                            dim=embed_dim,
                            num_heads=num_heads,
                            qkv_bias=True,
                            attn_drop=0.0,
                            proj_drop=0.0,
                        )

                        mlp_hidden = int(embed_dim * mlp_ratio)
                        mlp_mod = Mlp(
                            in_features=embed_dim,
                            hidden_features=mlp_hidden,
                            act_layer=nn.GELU,
                            drop=0.0,
                        )

                        for bs in bss:
                            # 逐层测
                            for layer_type in ["patch_embed", "attn", "mlp"]:
                                try:
                                    if layer_type == "patch_embed":
                                        input_shape = (bs, IN_CHANS, img, img)
                                        ms, peak_mem_mb = measure_module(
                                            patch_mod, input_shape, prec=prec,
                                            runs=20, warmup=8
                                        )
                                    elif layer_type == "attn":
                                        input_shape = (bs, L_eff, embed_dim)
                                        ms, peak_mem_mb = measure_module(
                                            attn_mod, input_shape, prec=prec,
                                            runs=20, warmup=8
                                        )
                                    else:  # mlp
                                        input_shape = (bs, L_eff, embed_dim)
                                        ms, peak_mem_mb = measure_module(
                                            mlp_mod, input_shape, prec=prec,
                                            runs=20, warmup=8
                                        )
                                    status = "ok"
                                except torch.cuda.OutOfMemoryError:
                                    ms = ""
                                    peak_mem_mb = ""
                                    status = "oom"
                                    print(
                                        f"[WARN] OOM at cfg={cfg['cfg']} "
                                        f"layer={layer_type} prec={prec} "
                                        f"keep={keep_ratio:.3f} img={img} bs={bs}",
                                        flush=True,
                                    )
                                    torch.cuda.empty_cache()

                                print(
                                    f"[{args.device} {cfg['cfg']} {prec}] "
                                    f"{layer_type} keep={keep_ratio:.3f} img={img} bs={bs} "
                                    f"→ status={status}, ms={ms}, mem={peak_mem_mb}",
                                    flush=True,
                                )

                                w.writerow(
                                    [
                                        args.device,
                                        cfg["cfg"],
                                        layer_type,
                                        prec,
                                        img,
                                        bs,
                                        f"{keep_ratio:.6f}",
                                        L_patch,
                                        L_eff,
                                        f"{float(ms):.6f}" if status == "ok" else "",
                                        f"{float(peak_mem_mb):.3f}" if status == "ok" else "",
                                        status,
                                        depth,
                                        embed_dim,
                                        num_heads,
                                        mlp_ratio,
                                        cfg["complexity_ratio"],
                                        cfg["head_dim"],
                                        cfg["tp_world_size"],
                                    ]
                                )

                        # 释放模块占用的显存
                        del patch_mod, attn_mod, mlp_mod
                        torch.cuda.empty_cache()

    print(f"\n[done] write layer dataset → {args.out_csv}")


if __name__ == "__main__":
    main()
