import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEURAGENIX_ROOT = ROOT.parent / "HeurAgenix"
sys.path.insert(0, str(HEURAGENIX_ROOT))
sys.path.insert(0, str(HEURAGENIX_ROOT / "src"))

from src.core import BaseEnv, BaseOperator, BaseSolution  # noqa: E402
from pipeline.hyper_heuristics import RandomHyperHeuristic  # noqa: E402


@dataclass
class DummySolution(BaseSolution):
    value: int


class DummyEnv(BaseEnv):
    def load_data(self, path: str):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def init_solution(self) -> DummySolution:
        return DummySolution(value=0)

    def get_key_value(self, solution: DummySolution) -> float:
        return float(solution.value)


class IncOperator(BaseOperator):
    def run(self, solution: DummySolution) -> DummySolution:
        return DummySolution(value=solution.value + 1)


def inc_heuristic(problem_state, algorithm_data):
    return IncOperator(), {}


def test_random_hyper_heuristic_runs():
    with tempfile.TemporaryDirectory() as d:
        data_path = Path(d) / "data.json"
        data_path.write_text("{}", encoding="utf-8")
        env = DummyEnv(str(data_path))
        rng = random.Random(0)
        hh = RandomHyperHeuristic(env, [inc_heuristic], rng, selection_frequency=1)
        hh.run(2)
        assert len(env.recordings) == 2
        assert all(rec["stage"] == "heuragenix_random_hh" for rec in env.recordings)
