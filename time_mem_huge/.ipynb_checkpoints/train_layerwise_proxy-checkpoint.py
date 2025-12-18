# train_layerwise_proxy.py

import argparse
import csv
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# 和 profiling 时一致
PATCH_SIZE = 16
IN_CHANS = 3


# ----------------- 基础 FLOPs / Bytes 估算 ----------------- #

def gemm_flops_bytes(M, K, N, bpe):
    flops = 2.0 * M * K * N
    bytes_ = bpe * (M * K + K * N + M * N)
    return flops, bytes_


def attn_core_flops_bytes(L, dmodel, dk, dv, heads, bpe):
    """自注意力核心 FLOPs / Bytes 近似"""
    H = heads if heads is not None else max(1, dmodel // dk)

    # FLOPs ~ 2 * H * L^2 * (dk + dv)
    flops = 2.0 * H * (L ** 2) * (dk + dv)

    # Bytes 近似：Q/K/V + 输出 + 注意力矩阵
    bytes_qkv = 3.0 * L * H * dk
    bytes_out = 1.0 * L * H * dv
    bytes_attn = 2.0 * (L ** 2) * H
    bytes_ = bpe * (bytes_qkv + bytes_out + bytes_attn)
    return flops, bytes_


def flops_bytes_layer(row, bpe):
    """
    给定一条 layer 样本（含 layer_type / embed_dim / num_heads / L_eff ...），
    估计该层的 FLOPs / Bytes（单样本）。
    """
    layer_type = row["layer_type"]     # patch_embed / attn / mlp
    bs = row["bs"]
    embed_dim = row["embed_dim"]
    num_heads = max(1, row["num_heads"])
    mlp_ratio = row["mlp_ratio"]
    L_patch = row["L_patch"]
    L_eff = row["L_eff"]

    if layer_type == "patch_embed":
        # conv/linear → [B, 3, H, W] -> [B, L_patch, C]
        M = L_patch * bs
        K = IN_CHANS * PATCH_SIZE * PATCH_SIZE
        N = embed_dim
        F, B = gemm_flops_bytes(M, K, N, bpe)

    elif layer_type == "attn":
        # QKV + attn_core + proj
        head_dim = embed_dim // num_heads
        dk = dv = head_dim

        # QKV GEMM
        M_qkv = L_eff * bs
        K_qkv = embed_dim
        N_qkv = 3 * embed_dim
        F_qkv, B_qkv = gemm_flops_bytes(M_qkv, K_qkv, N_qkv, bpe)

        # 注意力核心（per head）
        F_attn, B_attn = attn_core_flops_bytes(
            L=L_eff, dmodel=embed_dim, dk=dk, dv=dv,
            heads=num_heads, bpe=bpe
        )
        F_attn *= bs
        B_attn *= bs

        # 输出投影
        M_proj = L_eff * bs
        K_proj = embed_dim
        N_proj = embed_dim
        F_proj, B_proj = gemm_flops_bytes(M_proj, K_proj, N_proj, bpe)

        F = F_qkv + F_attn + F_proj
        B = B_qkv + B_attn + B_proj

    else:  # mlp
        d_ff = int(embed_dim * mlp_ratio)
        M = L_eff * bs
        F1, B1 = gemm_flops_bytes(M, embed_dim, d_ff, bpe)
        F2, B2 = gemm_flops_bytes(M, d_ff, embed_dim, bpe)
        F = F1 + F2
        B = B1 + B2

    return F, B


# ----------------- 数据加载 ----------------- #

def load_gpu_data(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    chip_map = {}
    for chip in data.get("chiplets", []):
        name = chip["name"]
        chip_map[name] = chip

    defaults = data.get("defaults", {})
    return chip_map, defaults


def load_layer_dataset(path):
    """
    读取 layer_dataset.csv，保持字段与你之前的 profile 输出一致。
    """
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "device": row["device"],
                    "cfg": row["cfg"],
                    "layer_type": row["layer_type"],  # patch_embed / attn / mlp
                    "prec": row["prec"],              # fp16 / fp32
                    "img": int(row["img"]),
                    "bs": int(row["bs"]),
                    "keep_ratio": float(row["keep_ratio"]),
                    "L_patch": int(row["L_patch"]),
                    "L_eff": int(row["L_eff"]),
                    "ms": row["ms"],                  # 可能为空
                    "peak_mem_mb": row["peak_mem_mb"],
                    "status": row["status"],          # ok / oom / ...
                    "depth": int(row["depth"]),
                    "embed_dim": int(row["embed_dim"]),
                    "num_heads": int(row["num_heads"]),
                    "mlp_ratio": float(row["mlp_ratio"]),
                    "complexity_ratio": float(row["complexity_ratio"]),
                    "head_dim": int(row["head_dim"]),
                    "tp_world_size": int(row["tp_world_size"]),
                }
            )
    return rows


# ----------------- 特征构造 ----------------- #

def build_features(rows, chip_map, defaults):
    """
    根据每条 layer 样本构建特征 X、目标 y_ms / y_mem，以及 meta 信息。
    特征中包含：
      - 层级 FLOPs / Bytes
      - 简单 roofline 下界 (t_roof, t_comp/t_roof, t_bw/t_roof)
      - img / bs / keep_ratio / L_patch / L_eff
      - depth/embed/heads/mlp_ratio/head_dim/complexity_ratio
      - 所在 GPU 的 peak_flops / peak_bw
      - layer_type one-hot
    """
    feats = []
    y_ms = []
    y_mem = []
    meta = []

    eta_comp_default = float(defaults.get("eta_comp", 0.75))
    eta_bw_default   = float(defaults.get("eta_bw", 0.75))

    for row in rows:
        if row["status"] != "ok":
            continue
        if row["ms"] == "" or row["peak_mem_mb"] == "":
            continue

        layer_type = row["layer_type"]
        prec = row["prec"].lower()
        if prec == "fp16":
            prec_tag = "FP16"
            bpe = 2
        else:
            prec_tag = "FP32"
            bpe = 4

        chip_key = f"{row['device']}_{prec_tag}"
        chip = chip_map.get(chip_key, None)
        if chip is None:
            print(f"[warn] skip sample: unknown chip {chip_key}")
            continue

        peak_flops = float(chip["peak_flops"])
        peak_bw = float(chip["peak_bw"])
        eta_comp = eta_comp_default
        eta_bw = eta_bw_default

        # 层级 FLOPs/Bytes
        F_L, B_L = flops_bytes_layer(row, bpe)

        # 简单 roofline 下界（不当最终 proxy，只作为输入特征）
        t_comp = F_L / (eta_comp * peak_flops + 1e-9)
        t_bw   = B_L / (eta_bw * peak_bw + 1e-9)
        t_roof = max(t_comp, t_bw)

        # layer_type one-hot
        if layer_type == "patch_embed":
            lt = [1.0, 0.0, 0.0]
        elif layer_type == "attn":
            lt = [0.0, 1.0, 0.0]
        else:  # mlp
            lt = [0.0, 0.0, 1.0]

        img = row["img"]
        bs = row["bs"]
        keep = row["keep_ratio"]
        L_patch = row["L_patch"]
        L_eff = row["L_eff"]
        embed_dim = row["embed_dim"]
        num_heads = max(1, row["num_heads"])
        mlp_ratio = row["mlp_ratio"]
        head_dim = row["head_dim"]

        x = [
            # 1) layer FLOPs / Bytes
            math.log10(F_L + 1e-9),
            math.log10(B_L + 1e-9),

            # 2) roofline 下界
            math.log10(t_roof + 1e-9),
            t_comp / (t_roof + 1e-9),
            t_bw / (t_roof + 1e-9),

            # 3) 输入规模 & token 信息
            img / 224.0,
            bs / 64.0,
            keep,
            L_patch / 256.0,
            L_eff / 256.0,

            # 4) 结构超参（相对 ViT-H）
            embed_dim / 1280.0,
            num_heads / 16.0,
            head_dim / 128.0,
            mlp_ratio / 4.0,
            row["complexity_ratio"],

            # 5) GPU 峰值参数
            math.log10(peak_flops + 1e-9),
            math.log10(peak_bw + 1e-9),

            # 6) layer 类型 one-hot
            *lt,
        ]

        feats.append(x)
        y_ms.append(float(row["ms"]))
        y_mem.append(float(row["peak_mem_mb"]))
        meta.append(
            {
                "cfg": row["cfg"],
                "prec": row["prec"],
                "device": row["device"],
                "layer_type": layer_type,
            }
        )

    X = np.asarray(feats, dtype=np.float32)
    y_ms = np.asarray(y_ms, dtype=np.float32)
    y_mem = np.asarray(y_mem, dtype=np.float32)
    return X, y_ms, y_mem, meta


def split_train_test_by_cfg(rows, train_ratio=0.8, seed=42):
    """
    按 cfg 划分 train/test，保证同一 cfg 的层不会泄露到 test。
    """
    cfgs = sorted({r["cfg"] for r in rows if r["status"] == "ok"})
    random.Random(seed).shuffle(cfgs)
    n_train = int(len(cfgs) * train_ratio)
    train_cfgs = set(cfgs[:n_train])
    test_cfgs  = set(cfgs[n_train:])

    train_rows, test_rows = [], []
    for r in rows:
        if r["status"] != "ok":
            continue
        if r["cfg"] in train_cfgs:
            train_rows.append(r)
        elif r["cfg"] in test_cfgs:
            test_rows.append(r)

    return train_rows, test_rows, train_cfgs, test_cfgs


# ----------------- 模型与指标 ----------------- #

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mape(pred, gt, eps=1e-6):
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    return float(np.mean(np.abs(pred - gt) / np.maximum(np.abs(gt), eps)))


def mape_group(pred, gt, metas, key_fn):
    buckets = defaultdict(lambda: {"pred": [], "gt": []})
    for p, g, m in zip(pred, gt, metas):
        k = key_fn(m)
        buckets[k]["pred"].append(p)
        buckets[k]["gt"].append(g)
    out = {}
    for k, v in buckets.items():
        out[k] = mape(v["pred"], v["gt"])
    return out


# ----------------- 训练 / 推理（log 目标） ----------------- #

def train_regressor(model, X_train, y_train, epochs=500, lr=1e-3,
                    weight_decay=1e-5, log_target=True, target_name="ms"):
    """
    log_target=True 时，以 log(y+eps) 为回归目标，更贴近相对误差优化。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    eps = 1e-3
    if log_target:
        y_proc = np.log(y_train + eps)
    else:
        y_proc = y_train

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_proc.astype(np.float32)).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    model.train()
    for e in range(1, epochs + 1):
        opt.zero_grad()
        pred = model(X_t)
        loss = mse(pred, y_t)
        loss.backward()
        opt.step()

        if e % max(1, epochs // 5) == 0:
            print(f"[train-{target_name}] epoch {e}/{epochs}, loss={loss.item():.6f}")
    return model


def predict_regressor(model, X, log_target=True):
    device = next(model.parameters()).device
    X_t = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()

    if log_target:
        eps = 1e-3
        pred = np.exp(pred) - eps
        pred = np.maximum(pred, 0.0)
    return pred


# ----------------- 主流程 ----------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="layer_dataset.csv")
    ap.add_argument("--gpu_yaml", default="gpu_data.yaml")
    ap.add_argument("--out", default="proxy_layer_report_separate.json")
    ap.add_argument("--save_ms", default="layer_proxy_ms.pt")
    ap.add_argument("--save_mem", default="layer_proxy_mem.pt")
    args = ap.parse_args()

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    chip_map, defaults = load_gpu_data(args.gpu_yaml)
    rows = load_layer_dataset(args.data)
    print(f"[data] total layer samples (raw) = {len(rows)}")

    train_rows, test_rows, train_cfgs, test_cfgs = split_train_test_by_cfg(rows, train_ratio=0.8)
    print(f"[split] train_cfgs = {len(train_cfgs)}, test_cfgs = {len(test_cfgs)}")
    print(f"[split] train_samples(raw) = {len(train_rows)}, test_samples(raw) = {len(test_rows)}")

    X_train, y_ms_train, y_mem_train, meta_train = build_features(train_rows, chip_map, defaults)
    X_test,  y_ms_test,  y_mem_test,  meta_test  = build_features(test_rows,  chip_map, defaults)

    print(f"[feat] feature_dim = {X_train.shape[1]}")
    print(f"[feat] train_samples = {X_train.shape[0]}, test_samples = {X_test.shape[0]}")

    in_dim = X_train.shape[1]

    # ---------- 1) 只拟合时间 ms 的代理 ----------
    print("\n=== Train time proxy (ms) ===")
    ms_model = MLPRegressor(in_dim=in_dim, hidden=128, depth=3)
    ms_model = train_regressor(
        ms_model, X_train, y_ms_train,
        epochs=1200, lr=1e-3, weight_decay=1e-5,
        log_target=True, target_name="ms"
    )

    ms_pred_train = predict_regressor(ms_model, X_train, log_target=True)
    ms_pred_test  = predict_regressor(ms_model, X_test,  log_target=True)

    ms_train_mape = mape(ms_pred_train, y_ms_train)
    ms_test_mape  = mape(ms_pred_test,  y_ms_test)

    ms_per_prec_train = mape_group(ms_pred_train, y_ms_train, meta_train, key_fn=lambda m: m["prec"])
    ms_per_prec_test  = mape_group(ms_pred_test,  y_ms_test,  meta_test,  key_fn=lambda m: m["prec"])
    ms_per_layer_test = mape_group(ms_pred_test,  y_ms_test,  meta_test,  key_fn=lambda m: m["layer_type"])

    print(f"[MS]  train MAPE = {ms_train_mape*100:.2f}%")
    print(f"[MS]  test  MAPE = {ms_test_mape*100:.2f}%")

    torch.save(ms_model.state_dict(), args.save_ms)
    print(f"[done] time proxy saved to {args.save_ms}")

    # ---------- 2) 只拟合显存 mem 的代理 ----------
    print("\n=== Train memory proxy (MB) ===")
    mem_model = MLPRegressor(in_dim=in_dim, hidden=128, depth=3)
    mem_model = train_regressor(
        mem_model, X_train, y_mem_train,
        epochs=1200, lr=1e-3, weight_decay=1e-5,
        log_target=True, target_name="mem"
    )

    mem_pred_train = predict_regressor(mem_model, X_train, log_target=True)
    mem_pred_test  = predict_regressor(mem_model, X_test,  log_target=True)

    mem_train_mape = mape(mem_pred_train, y_mem_train)
    mem_test_mape  = mape(mem_pred_test,  y_mem_test)

    mem_per_prec_train = mape_group(mem_pred_train, y_mem_train, meta_train, key_fn=lambda m: m["prec"])
    mem_per_prec_test  = mape_group(mem_pred_test,  y_mem_test,  meta_test,  key_fn=lambda m: m["prec"])
    mem_per_layer_test = mape_group(mem_pred_test, y_mem_test, meta_test,  key_fn=lambda m: m["layer_type"])

    print(f"[MEM] train MAPE = {mem_train_mape*100:.2f}%")
    print(f"[MEM] test  MAPE = {mem_test_mape*100:.2f}%")

    torch.save(mem_model.state_dict(), args.save_mem)
    print(f"[done] memory proxy saved to {args.save_mem}")

    # ---------- 3) 写报告 ----------
    report = {
        "ms": {
            "train_mape": ms_train_mape,
            "test_mape": ms_test_mape,
            "per_prec_train": ms_per_prec_train,
            "per_prec_test": ms_per_prec_test,
            "per_layer_type_test": ms_per_layer_test,
        },
        "mem": {
            "train_mape": mem_train_mape,
            "test_mape": mem_test_mape,
            "per_prec_train": mem_per_prec_train,
            "per_prec_test": mem_per_prec_test,
            "per_layer_type_test": mem_per_layer_test,
        },
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report written to {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
