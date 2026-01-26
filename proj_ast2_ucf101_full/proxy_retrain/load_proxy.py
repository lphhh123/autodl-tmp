# proxy_retrain/load_proxy.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from proxy_retrain.proxy_utils import build_design_matrix, expm1_safe, MLPRegressor


class ProxyInfer:
    def __init__(self, bundle: Dict, device: str = "cpu"):
        self.bundle = bundle
        self.device = device

        input_dim = int(bundle["input_dim"])
        hidden = bundle.get("model_hidden", [256, 256, 128])
        dropout = float(bundle.get("dropout", 0.1))

        model = MLPRegressor(input_dim=input_dim, hidden=hidden, dropout=dropout)
        model.load_state_dict(bundle["model_state"])
        model.to(device)
        model.eval()
        self.model = model

        self.num_cols = bundle["num_cols"]
        self.cat_cols = bundle["cat_cols"]
        self.cat_vocab = bundle["cat_vocab"]
        self.stats = bundle["stats"]
        self.target_mode = bundle["target_mode"]

    @classmethod
    def from_pt(cls, ckpt_path: str, device: str = "cpu") -> "ProxyInfer":
        bundle = torch.load(ckpt_path, map_location="cpu")
        return cls(bundle=bundle, device=device)

    @torch.no_grad()
    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_design_matrix(
            df,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            cat_vocab=self.cat_vocab,
            standardize=True,
            stats=self.stats,
        )
        xb = torch.from_numpy(X).to(self.device)
        pred = self.model(xb).detach().cpu().numpy()

        if self.target_mode == "log":
            raw = expm1_safe(pred, eps=1e-6)
            # 双保险：保证非负
            raw = np.maximum(raw, 1e-6)
            return raw
        return pred
