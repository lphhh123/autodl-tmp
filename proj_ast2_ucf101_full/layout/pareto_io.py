from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layout.pareto import ParetoSet


def write_pareto_points_csv(pareto: "ParetoSet", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comm_norm", "therm_norm", "total_scalar", "stage", "iter", "seed"])
        for p in pareto.points:
            meta = p.meta or {}
            w.writerow(
                [
                    float(p.comm_norm),
                    float(p.therm_norm),
                    float(meta.get("total_scalar", 0.0)),
                    str(meta.get("stage", "")),
                    int(meta.get("iter", -1)),
                    int(meta.get("seed", -1)),
                ]
            )
