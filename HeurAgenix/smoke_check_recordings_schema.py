import json
import tempfile
from pathlib import Path

from src.problems.base.env import BaseEnv


class DummySol:
    def __init__(self, assign):
        self.assign = list(assign)

    def get_solution_value(self):
        return 0.0


class DummyEnv(BaseEnv):
    def load_data(self, path: str):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def init_solution(self):
        seed = self.instance_data.get("seed", {}) or {}
        a = seed.get("assign_seed", [0, 1, 2])
        if not isinstance(a, (list, tuple)):
            a = [0, 1, 2]
        return DummySol(a)

    def get_key_value(self, solution):
        return 0.0

    def get_problem_state(self):
        return {}


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inst = {
            "seed": {"seed_id": 123, "assign_seed": [0, 1, 2]},
            "slots": {"S": 3, "tdp": [1, 1, 1]},
            "sites": {"Ns": 3, "sites_xy": [[0, 0], [1, 0], [0, 1]]},
            "mapping": {"traffic_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]},
        }
        inst_path = td / "case.json"
        inst_path.write_text(json.dumps(inst), encoding="utf-8")

        out_dir = td / "out"
        env = DummyEnv(str(inst_path))
        env._seed_id = 123
        env.seed = 123
        env.reset(output_dir=str(out_dir))

        rec_path = out_dir / "recordings.jsonl"
        assert rec_path.exists(), "recordings.jsonl not created"
        first = json.loads(rec_path.read_text(encoding="utf-8").splitlines()[0])
        assert "seed_id" in first, "seed_id missing in init record"
        assert "op_args_json" in first, "op_args_json missing"
        assert "signature" in first, "signature missing"
        print("[SMOKE] recordings schema OK:", first.keys())


if __name__ == "__main__":
    main()
