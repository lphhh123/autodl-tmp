"""HeurAgenix launch script for running hyper-heuristics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline.hyper_heuristics import LLMSelectionHyperHeuristic, RandomSelectionHyperHeuristic
from util.llm_client.get_llm_client import get_llm_client


def _search_file(filename: str, problem: str, folder: str = "data") -> Path | None:
    base = Path.cwd() / folder / problem
    direct = base / filename
    if direct.exists():
        return direct
    for candidate in base.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def _resolve_test_data(problem: str, test_name: str) -> Path:
    test_path = Path(test_name)
    if test_path.exists():
        return test_path
    candidate = _search_file(test_name, problem, folder="data")
    if candidate is not None:
        return candidate
    raise FileNotFoundError(f"test_data '{test_name}' not found under data/{problem}")


def _load_env(problem: str, data_path: str, rng_seed: int) -> object:
    module = __import__(f"problems.{problem}.env", fromlist=["Env"])
    env_cls = getattr(module, "Env")
    import random

    rng = random.Random(rng_seed)
    return env_cls(data_path, rng=rng)


def _resolve_heuristic_files(heuristic_dir: str, problem: str) -> list[str]:
    candidate = Path(heuristic_dir)
    if candidate.exists():
        root = candidate
    else:
        root = Path(__file__).resolve().parent / "src" / "problems" / problem / "heuristics" / heuristic_dir
    if not root.exists():
        raise FileNotFoundError(f"Heuristic directory not found: {root}")
    files = [str(path) for path in sorted(root.glob("*.py")) if not path.name.startswith("__")]
    if not files:
        raise RuntimeError(f"No heuristics found in {root}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem", required=True)
    parser.add_argument("-e", "--engine", required=True, choices=["llm_hh", "random_hh"])
    parser.add_argument("-d", "--heuristic_dir", required=True)
    parser.add_argument("-t", "--test_data", required=True)
    parser.add_argument("-l", "--llm_config", default=None)
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=1.0)
    parser.add_argument("-m", "--selection_frequency", type=int, default=5)
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=4)
    parser.add_argument("-b", "--rollout_budget", type=int, default=0)
    parser.add_argument("-r", "--result", default="result")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = args.problem
    test_path = _resolve_test_data(problem, args.test_data)
    env = _load_env(problem, str(test_path), args.seed)

    heuristic_files = _resolve_heuristic_files(args.heuristic_dir, problem)
    steps = max(1, int(float(args.iterations_scale_factor) * getattr(env, "problem_size", 1)))

    output_root = Path.cwd() / "output" / problem / test_path.name / args.result / args.engine
    output_root.mkdir(parents=True, exist_ok=True)
    usage_path = output_root / "llm_usage.jsonl"

    if args.engine == "llm_hh":
        if not args.llm_config:
            raise RuntimeError("llm_config is required for llm_hh")
        llm_client = get_llm_client(args.llm_config)
        if llm_client is None:
            raise RuntimeError("Failed to load LLM client config")
        hh = LLMSelectionHyperHeuristic(
            llm_client=llm_client,
            heuristic_pool=heuristic_files,
            problem=problem,
            iterations_scale_factor=args.iterations_scale_factor,
            selection_frequency=args.selection_frequency,
            num_candidate_heuristics=args.num_candidate_heuristics,
            rollout_budget=args.rollout_budget,
            rng=env.rng,
            usage_path=str(usage_path),
        )
    else:
        hh = RandomSelectionHyperHeuristic(
            heuristic_pool=heuristic_files,
            problem=problem,
            iterations_scale_factor=args.iterations_scale_factor,
            selection_frequency=args.selection_frequency,
            rng=env.rng,
        )

    hh.run(env)

    if getattr(hh, "usage_records", None) and args.engine != "llm_hh":
        with usage_path.open("w", encoding="utf-8") as f:
            for record in hh.usage_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    env.dump_result({"engine": args.engine, "steps": steps}, str(output_root))


if __name__ == "__main__":
    main()
