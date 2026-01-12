from pathlib import Path
import json
import numpy as np


def gen_case(seed=1, S=8, Ns=16):
    np.random.seed(seed)
    data = {
        "chiplets": {"S": S, "tdp": np.random.uniform(5, 15, S).tolist()},
        "sites": {"Ns": Ns, "sites_xy": np.random.uniform(-1, 1, (Ns, 2)).tolist()},
        "mapping": {"traffic_matrix": np.random.rand(S, S).tolist()},
        "weights": {"w_comm": 0.5, "w_therm": 0.5},
        "sigma": 1.0,
        "baseline": {"L_comm": 1.0, "L_therm": 1.0},
        "wafer": {"radius": 1.0},
        "assign_seed": seed,
    }
    return data


def main():
    outdir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "wafer_layout"
        / "test_data"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    for s in [1, 2, 3]:
        data = gen_case(seed=s)
        (outdir / f"case_seed{s}.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
