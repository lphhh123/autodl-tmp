from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from hw_proxy.layer_proxy_model import LayerProxyModel


@dataclass
class CalibRow:
    feat: List[float]
    lat_ms: float
    mem_mb: float
    power_w: float


def _read_csv_rows(path: Path) -> List[CalibRow]:
    """
    Expect header contains at least:
      feat_0..feat_{D-1}, lat_ms, mem_mb, power_w
    """
    rows: List[CalibRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {path}")
        # infer D
        feat_cols = [c for c in reader.fieldnames if c.startswith("feat_")]
        feat_cols = sorted(feat_cols, key=lambda x: int(x.split("_")[1]))
        if not feat_cols:
            raise ValueError(f"CSV must contain feat_0..feat_D columns, got fields={reader.fieldnames}")

        for r in reader:
            feat = [float(r[c]) for c in feat_cols]
            lat = float(r.get("lat_ms", r.get("latency_ms", 0.0)))
            mem = float(r.get("mem_mb", 0.0))
            pwr = float(r.get("power_w", 0.0))
            rows.append(CalibRow(feat=feat, lat_ms=lat, mem_mb=mem, power_w=pwr))
    return rows


def _train_one(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(epochs)):
        pred = model(x).squeeze(-1)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_layer_proxies_from_csv(
    calib_csv: str,
    out_dir: str,
    in_dim: Optional[int] = None,
    device: str = "cpu",
    epochs: int = 200,
    lr: float = 1e-3,
) -> Dict[str, str]:
    """
    Train 3 LayerProxyModel for latency/mem/power.
    Save:
      latency_proxy.pth, mem_proxy.pth, power_proxy.pth
    Return saved paths.
    """
    p = Path(calib_csv)
    if not p.is_file():
        raise FileNotFoundError(f"Calibration CSV not found: {p}")

    rows = _read_csv_rows(p)
    if not rows:
        raise ValueError(f"No rows in calibration CSV: {p}")

    D = len(rows[0].feat) if in_dim is None else int(in_dim)
    X = torch.tensor([r.feat[:D] for r in rows], dtype=torch.float32, device=device)
    y_lat = torch.tensor([r.lat_ms for r in rows], dtype=torch.float32, device=device)
    y_mem = torch.tensor([r.mem_mb for r in rows], dtype=torch.float32, device=device)
    y_pow = torch.tensor([r.power_w for r in rows], dtype=torch.float32, device=device)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    lat_model = LayerProxyModel(D).to(device)
    mem_model = LayerProxyModel(D).to(device)
    pow_model = LayerProxyModel(D).to(device)

    lat_model = _train_one(X, y_lat, lat_model, epochs=epochs, lr=lr)
    mem_model = _train_one(X, y_mem, mem_model, epochs=epochs, lr=lr)
    pow_model = _train_one(X, y_pow, pow_model, epochs=epochs, lr=lr)

    lat_path = outp / "latency_proxy.pth"
    mem_path = outp / "mem_proxy.pth"
    pow_path = outp / "power_proxy.pth"

    torch.save(lat_model.state_dict(), lat_path)
    torch.save(mem_model.state_dict(), mem_path)
    torch.save(pow_model.state_dict(), pow_path)

    return {
        "latency_proxy": str(lat_path),
        "mem_proxy": str(mem_path),
        "power_proxy": str(pow_path),
        "num_rows": str(len(rows)),
        "in_dim": str(D),
    }
