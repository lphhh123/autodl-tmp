"""HeurAgenix launch script for running hyper-heuristics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
from pipeline.hyper_heuristics.random import RandomHyperHeuristic
from util.llm_client.get_llm_client import get_llm_client
from util.util import search_file


def _resolve_test_data(problem: str, test_name: str) -> Path:
    test_path = Path(test_name)
    if test_path.exists():
        return test_path
    candidate = search_file(test_name, problem)
    if candidate:
        return Path(candidate)
    raise FileNotFoundError(f"test_data '{test_name}' not found under data/{problem}")


def _load_env(problem: str, data_path: str, rng_seed: int) -> object:
    module = __import__(f"src.problems.{problem}.env", fromlist=["Env"])
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


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument(
        "--llm_timeout_s",
        type=float,
        default=None,
        help="Per LLM call timeout (seconds). None = client default.",
    )
    parser.add_argument(
        "--max_llm_failures",
        type=int,
        default=5,
        help="Max LLM failures before disabling LLM and falling back.",
    )
    parser.add_argument(
        "--fallback_mode",
        type=str,
        default="random",
        choices=["random", "disable_llm", "abort"],
        help="On LLM failure: choose random heuristic / disable llm permanently / abort run.",
    )
    args = parser.parse_args()

    problem = args.problem
    test_path = _resolve_test_data(problem, args.test_data)
    env = _load_env(problem, str(test_path), args.seed)

    heuristic_files = _resolve_heuristic_files(args.heuristic_dir, problem)
    steps = max(1, int(float(args.iterations_scale_factor) * getattr(env, "construction_steps", 1)))

    output_root = Path(os.environ.get("AMLT_OUTPUT_DIR", "output"))
    base_output_root = output_root / problem / args.result / test_path.name

    engine = args.engine
    llm_error = None
    llm_client = None
    if engine == "llm_hh":
        if not args.llm_config:
            llm_error = "llm_config is required for llm_hh"
        else:
            try:
                llm_client = get_llm_client(args.llm_config)
            except Exception as exc:  # noqa: BLE001
                llm_error = repr(exc)
        if llm_client is None:
            llm_error = llm_error or "llm_client_unavailable"
            engine = "random_hh"
    output_root = base_output_root / f"seed{args.seed}_{engine}"
    output_root.mkdir(parents=True, exist_ok=True)
    usage_path = output_root / "llm_usage.jsonl"

    initial_usage: List[Dict] = []
    if engine != args.engine:
        initial_usage.append(
            {"ok": False, "engine": "llm_hh", "reason": "llm_unavailable", "error": llm_error}
        )
    if initial_usage:
        _write_jsonl(usage_path, initial_usage)
    if engine == "llm_hh":
        hh = LLMSelectionHyperHeuristic(
            llm_client=llm_client,
            heuristic_pool=heuristic_files,
            problem=problem,
            iterations_scale_factor=args.iterations_scale_factor,
            selection_frequency=args.selection_frequency,
            num_candidate_heuristics=args.num_candidate_heuristics,
            rollout_budget=args.rollout_budget,
            llm_timeout_s=args.llm_timeout_s,
            max_llm_failures=args.max_llm_failures,
            fallback_mode=args.fallback_mode,
            rng=env.rng,
            usage_path=str(usage_path),
            llm_error=llm_error,
        )
    else:
        hh = RandomHyperHeuristic(
            heuristic_pool=heuristic_files,
            problem=problem,
            iterations_scale_factor=args.iterations_scale_factor,
            selection_frequency=args.selection_frequency,
            rng=env.rng,
        )

    env.reset(str(output_root))
    hh.run(env)

    if not usage_path.exists():
        records: List[Dict] = []
        if engine == "llm_hh":
            records = list(getattr(hh, "usage_records", []) or [])
        elif engine == "random_hh":
            records = list(getattr(hh, "usage_records", []) or [])
            if not records:
                records = [{"ok": False, "engine": "random_hh"}]
        if not records:
            records = [{"ok": False, "engine": engine, "reason": "missing_usage_records"}]
        _write_jsonl(usage_path, initial_usage + records)

    env.dump_result()


if __name__ == "__main__":
    main()
