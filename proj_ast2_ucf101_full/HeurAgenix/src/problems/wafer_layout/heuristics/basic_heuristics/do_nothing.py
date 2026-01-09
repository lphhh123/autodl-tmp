from __future__ import annotations

from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoOp


def do_nothing(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[NoOp, Dict]:
    return NoOp(), {"noop": True}
