import os
import argparse
import random
from pathlib import Path

import numpy as np

from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.util import load_function


def _list_problems():
    problems_dir = Path(__file__).parent / "src" / "problems"
    out = []
    for d in problems_dir.iterdir():
        if d.is_dir() and d.name not in ["base", "__pycache__"]:
            out.append(d.name)
    return sorted(out)


def _resolve_test_data_dir(problem: str) -> Path:
    """
    Wrapper sets AMLT_DATA_DIR=<work_dir>/data and writes:
      <AMLT_DATA_DIR>/{problem}/test_data/*.json
    But some HeurAgenix utilities also use .../{problem}/data/test_data.
    We support both.
    """
    data_root = Path(os.environ.get("AMLT_DATA_DIR", "data")).resolve()
    cand1 = data_root / problem / "test_data"
    cand2 = data_root / problem / "data" / "test_data"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    cand1.mkdir(parents=True, exist_ok=True)
    return cand1


def _resolve_output_base() -> Path:
    """
    Wrapper sets AMLT_OUTPUT_DIR=<out_dir>/heuragenix_internal.
    We must write into <AMLT_OUTPUT_DIR>/output/... (NOT ../../output).
    """
    base = Path(os.environ.get("AMLT_OUTPUT_DIR", ".")).resolve()
    return base / "output"


def _import_env(problem: str):
    mod = __import__(f"src.problems.{problem}.env", fromlist=["Env"])
    return getattr(mod, "Env")


def parse_arguments():
    problems = _list_problems()

    parser = argparse.ArgumentParser(description="Launch Hyper-Heuristic")
    parser.add_argument("-p", "--problem", type=str, required=True, choices=problems)
    parser.add_argument(
        "-e", "--heuristic", type=str, required=True,
        help="heuristic function name OR llm_hh/random_hh"
    )
    parser.add_argument("-d", "--heuristic_dir", type=str, default="basic_heuristics")
    parser.add_argument("-t", "--test_data", type=str, default=None,
                        help="Comma-separated test file names. Default: all in test_data dir.")
    parser.add_argument("-l", "--llm_config_file", type=str, default=None)
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0)
    parser.add_argument("-m", "--selection_frequency", type=int, default=5)
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1)
    parser.add_argument("-b", "--rollout_budget", type=int, default=0)

    parser.add_argument("-r", "--result_dir", type=str, default="result")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--llm_timeout_s", type=int, default=30)
    parser.add_argument("--max_llm_failures", type=int, default=2)
    parser.add_argument("--fallback_on_llm_failure", type=str, default="random_hh",
                        choices=["random_hh", "stop"])

    parser.add_argument("-res", "--result_name", type=str, default=None)
    parser.add_argument("-exp", "--experiment_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)

    test_data_dir = _resolve_test_data_dir(args.problem)
    output_base = _resolve_output_base()

    if args.test_data is None or args.test_data.strip() == "":
        data_name_list = sorted([p.name for p in test_data_dir.iterdir() if p.is_file()])
    else:
        data_name_list = [x.strip() for x in args.test_data.split(",") if x.strip()]

    heur_dir = Path("src") / "problems" / args.problem / "heuristics" / args.heuristic_dir
    if not heur_dir.exists():
        raise FileNotFoundError(f"heuristic_dir not found: {heur_dir.resolve()}")

    heuristic_pool_files = [
        f.stem
        for f in heur_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("_") and f.name != "__init__.py"
    ]

    engine = args.heuristic
    if args.experiment_name:
        engine = args.experiment_name

    Env = _import_env(args.problem)

    heur_name = args.heuristic[:-3] if str(args.heuristic).endswith(".py") else args.heuristic

    for data_name in data_name_list:
        data_path = (test_data_dir / data_name).resolve()
        env = Env(str(data_path))

        problem_size = int(getattr(env, "problem_size", None) or getattr(env, "S", None) or 1)
        if args.max_steps is not None:
            env.max_steps = int(args.max_steps)
        else:
            env.max_steps = max(1, int(float(args.iterations_scale_factor) * max(1, problem_size)))

        env.llm_timeout_s = int(args.llm_timeout_s)
        env.max_llm_failures = int(args.max_llm_failures)
        env.fallback_on_llm_failure = str(args.fallback_on_llm_failure)
        seed_val = int(getattr(env, "seed", args.seed))

        case_stem = Path(data_name).stem
        out_dir = (output_base / args.problem / case_stem / args.result_dir / engine).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        usage_path = out_dir / "llm_usage.jsonl"
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        if not usage_path.exists():
            usage_path.write_text("", encoding="utf-8")
        env.reset(output_dir=str(out_dir))

        if args.heuristic == "random_hh":
            runner = RandomHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                iterations_scale_factor=float(args.iterations_scale_factor),
                heuristic_dir=str(heur_dir),
                selection_frequency=int(args.selection_frequency),
                seed=int(args.seed),
            )
        elif args.heuristic == "llm_hh":
            runner = LLMSelectionHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                heuristic_dir=str(heur_dir),
                llm_config_file=args.llm_config_file,
                iterations_scale_factor=float(args.iterations_scale_factor),
                selection_frequency=int(args.selection_frequency),
                num_candidate_heuristics=int(args.num_candidate_heuristics),
                rollout_budget=int(args.rollout_budget),
                output_dir=str(out_dir),
                seed=seed_val,
                llm_timeout_s=int(args.llm_timeout_s),
                max_llm_failures=int(args.max_llm_failures),
                fallback_on_llm_failure=str(args.fallback_on_llm_failure),
            )
        else:
            fn = load_function(heur_name, problem=args.problem)
            runner = SingleHyperHeuristic(
                heuristic=fn,
                iterations_scale_factor=float(args.iterations_scale_factor),
                output_dir=str(out_dir),
                seed=seed_val,
            )

        runner.run(env)
        env.dump_result()


if __name__ == "__main__":
    main()
