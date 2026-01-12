from __future__ import annotations

from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoopOperator


def do_nothing(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[NoopOperator, Dict]:
    return NoopOperator(), {}
