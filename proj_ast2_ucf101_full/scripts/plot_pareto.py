"""Plot Pareto scatter from layout_best.json (SPEC v4.3.2 §12.3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    best = json.loads(Path(args.best).read_text())
    pareto = best.get("pareto_front", [])
    if not pareto:
        raise SystemExit("No pareto_front found in best file")
    comm = [p["comm_norm"] for p in pareto]
    therm = [p["therm_norm"] for p in pareto]
    plt.figure(figsize=(6, 4))
    plt.scatter(comm, therm, s=16, c="blue", label="Pareto")
    knee = best.get("best", {}).get("objectives", {})
    if knee:
        plt.scatter([knee.get("comm_norm", 0)], [knee.get("therm_norm", 0)], c="red", marker="x", s=40, label="knee")
    plt.xlabel("comm_norm")
    plt.ylabel("therm_norm")
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
