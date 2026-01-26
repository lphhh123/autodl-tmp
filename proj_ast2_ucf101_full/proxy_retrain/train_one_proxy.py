# proxy_retrain/train_one_proxy.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from proxy_retrain.proxy_utils import (
    seed_everything,
    split_train_val_test,
    detect_feature_columns,
    build_design_matrix,
    MLPRegressor,
    huber_loss,
    compute_reg_metrics,
    save_bundle,
    save_torch_bundle,
    log1p_safe,
    expm1_safe,
)


def _prepare_xy(df: pd.DataFrame, target_col: str, target_mode: str) -> np.ndarray:
    y = df[target_col].astype(float).values
    if target_mode == "log":
        return log1p_safe(y, eps=1e-6).astype(np.float32)
    elif target_mode == "raw":
        return y.astype(np.float32)
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")


@torch.no_grad()
def _predict(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    pred = model(xb).detach().cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--target_col", type=str, required=True)  # e.g., ms_event / peak_mem_mb / energy_mj / avg_power_w
    ap.add_argument("--target_mode", type=str, default="log", choices=["log", "raw"])
    ap.add_argument("--nonneg_output", action="store_true", help="If set, enforce nonneg at inference via log-space or softplus in adapter.")

    ap.add_argument("--drop_cols", type=str, default="", help="comma-separated columns to drop (ids, paths, etc.)")
    ap.add_argument("--split_keys", type=str, default="device,cfg,layer_type,prec,img,bs,keep_ratio,L_eff,depth,embed_dim,num_heads,mlp_ratio,complexity_ratio,tp_world_size")
    ap.add_argument("--seed", type=int, default=2024)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=str, default="256,256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=12)

    args = ap.parse_args()
    seed_everything(args.seed)

    df = pd.read_csv(args.csv)
    assert args.target_col in df.columns, f"target_col {args.target_col} not in columns"

    # 清理：只保留 target 非空、非负（你如果要容忍 0，可以自己改）
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    # 对于要 log 的目标，必须 > -eps；一般真实测量不会为负
    if args.target_mode == "log":
        df = df[df[args.target_col].astype(float) > -1e-9].reset_index(drop=True)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    split_keys = [c.strip() for c in args.split_keys.split(",") if c.strip() and c in df.columns]

    # 划分
    split = split_train_val_test(df, seed=args.seed, split_key_cols=split_keys)
    train_df, val_df, test_df = split.train, split.val, split.test

    # 自动特征列
    num_cols, cat_cols = detect_feature_columns(
        train_df,
        target_cols=[args.target_col],
        drop_cols=drop_cols,
    )

    # build X
    X_train, bundle_X = build_design_matrix(train_df, num_cols, cat_cols, cat_vocab=None, standardize=True, stats=None)
    X_val, _ = build_design_matrix(val_df, num_cols, cat_cols, cat_vocab=bundle_X["cat_vocab"], standardize=True, stats=bundle_X["stats"])
    X_test, _ = build_design_matrix(test_df, num_cols, cat_cols, cat_vocab=bundle_X["cat_vocab"], standardize=True, stats=bundle_X["stats"])

    y_train = _prepare_xy(train_df, args.target_col, args.target_mode)
    y_val = _prepare_xy(val_df, args.target_col, args.target_mode)
    y_test = _prepare_xy(test_df, args.target_col, args.target_mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MLPRegressor(input_dim=bundle_X["input_dim"], hidden=hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = 1e30
    best_state = None
    bad = 0

    # dataloader (simple numpy batching)
    n = len(X_train)
    idx = np.arange(n)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)

        losses = []
        for s in range(0, n, args.batch):
            b = idx[s : s + args.batch]
            xb = torch.from_numpy(X_train[b]).to(device)
            yb = torch.from_numpy(y_train[b]).to(device)

            pred = model(xb)
            loss = huber_loss(pred, yb, delta=1.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        # val
        model.eval()
        pred_val = _predict(model, X_val, device=device)
        val_loss = float(np.mean((pred_val - y_val) ** 2))
        train_loss = float(np.mean(losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"[ep {ep:03d}] train_loss={train_loss:.6f} val_mse={val_loss:.6f} best={best_val:.6f} bad={bad}/{args.patience}")
        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试评估（回到 raw 空间再算 MAPE/排序）
    pred_train = _predict(model, X_train, device=device)
    pred_val = _predict(model, X_val, device=device)
    pred_test = _predict(model, X_test, device=device)

    def to_raw(pred_log: np.ndarray) -> np.ndarray:
        if args.target_mode == "log":
            return expm1_safe(pred_log, eps=1e-6)
        return pred_log

    y_train_raw = train_df[args.target_col].astype(float).values
    y_val_raw = val_df[args.target_col].astype(float).values
    y_test_raw = test_df[args.target_col].astype(float).values

    p_train_raw = to_raw(pred_train)
    p_val_raw = to_raw(pred_val)
    p_test_raw = to_raw(pred_test)

    # 非物理值检查（要求你论文里“0 容忍”）
    nonphys = {
        "train_count_le_0": int(np.sum(p_train_raw <= 0.0)),
        "val_count_le_0": int(np.sum(p_val_raw <= 0.0)),
        "test_count_le_0": int(np.sum(p_test_raw <= 0.0)),
        "train_min": float(np.min(p_train_raw)) if len(p_train_raw) else None,
        "val_min": float(np.min(p_val_raw)) if len(p_val_raw) else None,
        "test_min": float(np.min(p_test_raw)) if len(p_test_raw) else None,
    }

    rep = {
        "csv": args.csv,
        "target_col": args.target_col,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "features": {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "input_dim": int(bundle_X["input_dim"]),
        },
        "metrics": {
            "train": compute_reg_metrics(y_train_raw, p_train_raw),
            "val": compute_reg_metrics(y_val_raw, p_val_raw),
            "test": compute_reg_metrics(y_test_raw, p_test_raw),
        },
        "nonphysical": nonphys,
        "notes": [
            "MAPE 是必要但不充分：请重点看 spearman/pairwise_acc。",
            "若出现 <=0 的预测，请优先使用 log-space（本脚本默认）并检查数据与特征范围。",
        ],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    rep_path = os.path.join(args.out_dir, f"report_{args.target_col}.json")
    save_bundle(rep_path, rep)

    # 保存 torch bundle（你后续在主工程里 load 用）
    ckpt_path = os.path.join(args.out_dir, f"proxy_{args.target_col}.pt")
    torch_bundle = {
        "model_state": model.state_dict(),
        "model_hidden": hidden,
        "dropout": args.dropout,
        "input_dim": bundle_X["input_dim"],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_vocab": bundle_X["cat_vocab"],
        "stats": bundle_X["stats"],
        "target_col": args.target_col,
        "target_mode": args.target_mode,
    }
    save_torch_bundle(ckpt_path, torch_bundle)

    print(f"[OK] saved report: {rep_path}")
    print(f"[OK] saved ckpt  : {ckpt_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
