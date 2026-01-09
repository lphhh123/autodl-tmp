# HeurAgenix/src/pipeline/hyper_heuristic.py
import os
from pathlib import Path
from typing import List, Optional

from src.util.util import load_function
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic

import importlib


def _load_env(problem: str, data_name: str):
    mod = importlib.import_module(f"src.problems.{problem}.env")
    Env = getattr(mod, "Env")
    return Env(data_name)


def launch_heuristic(
    problem: str,
    test_data_dir: Path,
    data_name_list: List[str],
    output_base_dir: Path,
    heuristic_name: str,
    iterations_scale_factor: float,
    result_dir: str = "result",
):
    """
    Run a single heuristic function (not llm_hh/random_hh).
    """
    for data_name in data_name_list:
        env = _load_env(problem, data_name)
        # IMPORTANT: make steps effective
        if getattr(env, "problem_size", None) is None:
            env.problem_size = getattr(env, "S", 1)
        env.max_steps = max(1, int(float(iterations_scale_factor) * max(1, int(env.problem_size))))
        env.construction_steps = 0

        # output dir = output/{problem}/{test_data}/{result}/{heuristic_name}
        test_stem = Path(data_name).stem
        out_dir = output_base_dir / problem / test_stem / result_dir / heuristic_name
        os.makedirs(out_dir, exist_ok=True)
        env.reset(output_dir=str(out_dir))

        fn = load_function(heuristic_name, problem=problem)
        runner = SingleHyperHeuristic(fn, iterations_scale_factor=float(iterations_scale_factor), output_dir=str(out_dir))
        runner.run(env)
        env.dump_result()


def launch_heuristic_selector(
    problem: str,
    test_data_dir: Path,
    data_name_list: List[str],
    output_base_dir: Path,
    engine_name: str,
    heuristic_dir: str,
    llm_config_file: Optional[str],
    iterations_scale_factor: float,
    selection_frequency: int,
    num_candidate_heuristics: int,
    rollout_budget: int,
    result_dir: str = "result",
):
    """
    Run llm_hh / random_hh.
    Output layout MUST follow README:
      output/{problem}/{test_data}/{result}/{engine}/...
    """
    for data_name in data_name_list:
        env = _load_env(problem, data_name)
        if getattr(env, "problem_size", None) is None:
            env.problem_size = getattr(env, "S", 1)
        env.max_steps = max(1, int(float(iterations_scale_factor) * max(1, int(env.problem_size))))
        env.construction_steps = 0

        test_stem = Path(data_name).stem
        out_dir = output_base_dir / problem / test_stem / result_dir / engine_name
        os.makedirs(out_dir, exist_ok=True)
        env.reset(output_dir=str(out_dir))

        if engine_name == "random_hh":
            runner = RandomHyperHeuristic(
                heuristic_pool=None,
                problem=problem,
                iterations_scale_factor=float(iterations_scale_factor),
                heuristic_dir=heuristic_dir,
            )
        else:
            runner = LLMSelectionHyperHeuristic(
                heuristic_pool=None,
                problem=problem,
                heuristic_dir=heuristic_dir,
                llm_config_file=llm_config_file,
                iterations_scale_factor=float(iterations_scale_factor),
                selection_frequency=int(selection_frequency),
                num_candidate_heuristics=int(num_candidate_heuristics),
                rollout_budget=int(rollout_budget),
                output_dir=str(out_dir),
            )

        runner.run(env)
        env.dump_result()
