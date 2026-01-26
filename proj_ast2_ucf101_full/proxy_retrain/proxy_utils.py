# proxy_retrain/proxy_utils.py
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.stats import spearmanr, kendalltau
except Exception:
    spearmanr = None
    kendalltau = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 可复现：你也可以按需要打开/关闭
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _hash_split_ids(df: pd.DataFrame, key_cols: List[str], seed: int = 2024) -> np.ndarray:
    """
    给每行生成一个稳定的伪随机数，用于可复现划分。
    """
    if not key_cols:
        idx = np.arange(len(df))
        rs = np.random.RandomState(seed)
        rs.shuffle(idx)
        return idx.astype(np.float64) / max(1, len(df) - 1)

    # 组合字符串做 hash
    keys = df[key_cols].astype(str).agg("|".join, axis=1).values
    # Python hash 不稳定，自己做一个稳定 hash
    h = np.array([sum((i + 1) * ord(ch) for i, ch in enumerate(s)) for s in keys], dtype=np.int64)
    # 线性同余混洗
    x = (h ^ (seed * 1315423911)) & 0xFFFFFFFF
    x = (1103515245 * x + 12345) & 0x7FFFFFFF
    return x.astype(np.float64) / float(0x7FFFFFFF)


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_key_cols: Optional[List[str]] = None,
    seed: int = 2024,
) -> SplitData:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    u = _hash_split_ids(df, split_key_cols or [], seed=seed)
    train_mask = u < train_ratio
    val_mask = (u >= train_ratio) & (u < train_ratio + val_ratio)
    test_mask = u >= (train_ratio + val_ratio)
    return SplitData(
        train=df[train_mask].reset_index(drop=True),
        val=df[val_mask].reset_index(drop=True),
        test=df[test_mask].reset_index(drop=True),
    )


def detect_feature_columns(
    df: pd.DataFrame,
    target_cols: List[str],
    drop_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    自动找特征列：
      - numeric：除 target/drop 以外的数值列
      - cat：除 target/drop 以外的 object/bool 列
    """
    drop_cols = drop_cols or []
    ban = set(target_cols + drop_cols)

    num_cols = []
    cat_cols = []
    for c in df.columns:
        if c in ban:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def build_design_matrix(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    cat_vocab: Optional[Dict[str, List[str]]] = None,
    standardize: bool = True,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    数值列：可选标准化
    类别列：one-hot（使用 train 的 vocab 固定）
    """
    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    if num_cols:
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # cat vocab
    if cat_vocab is None:
        cat_vocab = {}
        for c in cat_cols:
            vs = sorted(df[c].astype(str).fillna("NA").unique().tolist())
            cat_vocab[c] = vs

    X_cat_parts = []
    for c in cat_cols:
        vs = cat_vocab[c]
        s = df[c].astype(str).fillna("NA")
        # one-hot
        m = np.zeros((len(df), len(vs)), dtype=np.float32)
        v2i = {v: i for i, v in enumerate(vs)}
        for i, v in enumerate(s.values):
            j = v2i.get(v, None)
            if j is not None:
                m[i, j] = 1.0
        X_cat_parts.append(m)

    X_cat = np.concatenate(X_cat_parts, axis=1) if X_cat_parts else np.zeros((len(df), 0), dtype=np.float32)

    # standardize numeric
    if stats is None:
        stats = {}
        if standardize and num_cols:
            mu = X_num.mean(axis=0).to_dict()
            sd = X_num.std(axis=0).replace(0.0, 1.0).to_dict()
            stats["num_mean"] = mu
            stats["num_std"] = sd
        else:
            stats["num_mean"] = {c: 0.0 for c in num_cols}
            stats["num_std"] = {c: 1.0 for c in num_cols}

    if num_cols:
        mu = np.array([stats["num_mean"][c] for c in num_cols], dtype=np.float32)
        sd = np.array([stats["num_std"][c] for c in num_cols], dtype=np.float32)
        Xn = X_num.values.astype(np.float32)
        if standardize:
            Xn = (Xn - mu) / sd
    else:
        Xn = np.zeros((len(df), 0), dtype=np.float32)

    X = np.concatenate([Xn, X_cat], axis=1).astype(np.float32)

    bundle = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_vocab": cat_vocab,
        "stats": stats,
        "input_dim": int(X.shape[1]),
    }
    return X, bundle


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] = [256, 256, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=delta)


@torch.no_grad()
def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    eps = 1e-12
    ape = np.abs(y_pred - y_true) / (np.abs(y_true) + eps)
    mape = float(np.mean(ape))
    p95 = float(np.quantile(ape, 0.95))
    p99 = float(np.quantile(ape, 0.99))
    mx = float(np.max(ape))

    out = {
        "mape": mape,
        "p95_ape": p95,
        "p99_ape": p99,
        "max_ape": mx,
    }

    # 排序一致性
    if spearmanr is not None:
        out["spearman_rho"] = float(spearmanr(y_true, y_pred).correlation)
    else:
        out["spearman_rho"] = None

    if kendalltau is not None:
        out["kendall_tau"] = float(kendalltau(y_true, y_pred).correlation)
    else:
        out["kendall_tau"] = None

    # Pairwise ordering accuracy（随机抽样）
    n = len(y_true)
    if n >= 2:
        rng = np.random.RandomState(0)
        pairs = min(50000, n * 10)
        i = rng.randint(0, n, size=pairs)
        j = rng.randint(0, n, size=pairs)
        mask = i != j
        i, j = i[mask], j[mask]
        yt = y_true[i] < y_true[j]
        yp = y_pred[i] < y_pred[j]
        out["pairwise_acc"] = float(np.mean(yt == yp))
    else:
        out["pairwise_acc"] = None

    return out


def save_bundle(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_torch_bundle(path: str, bundle: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(bundle, path)


def log1p_safe(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log(x + eps)


def expm1_safe(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # inverse of log(x+eps): exp(y)-eps
    return np.exp(x) - eps
