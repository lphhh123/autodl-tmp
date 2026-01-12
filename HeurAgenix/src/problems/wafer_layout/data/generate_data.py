from pathlib import Path
import numpy as np
import json


def gen_case(seed=1, S=8, Ns=16):
    np.random.seed(seed)
    return {
        "chiplets": {"S": S, "tdp": np.random.uniform(5, 15, S).tolist()},
        "sites": {"Ns": Ns, "sites_xy": np.random.uniform(-1, 1, (Ns, 2)).tolist()},
        "mapping": {"traffic_matrix": np.random.rand(S, S).tolist()},
        "weights": {"w_comm": 0.5, "w_therm": 0.5},
        "sigma": 1.0,
        "wafer": {"radius": 1.0},
        "assign_seed": seed,
    }


def main():
    base = Path(__file__).resolve().parents[2] / "data" / "wafer_layout" / "test_data"
    base.mkdir(parents=True, exist_ok=True)
    for s in range(3):
        case = gen_case(seed=s + 1)
        (base / f"case_seed{s + 1}.json").write_text(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
