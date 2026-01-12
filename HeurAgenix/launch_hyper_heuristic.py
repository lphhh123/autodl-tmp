import os
import sys
import argparse
import random
import importlib
from pathlib import Path

# make runnable from anywhere
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.util.output_dir import get_output_dir
from src.util.data_path import get_data_path
from src.util.util import load_function, get_heuristic_names
from src.util.llm_client.get_llm_client import get_llm_client

from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic


def _load_env(problem: str, data_name: str):
    mod = importlib.import_module(f"src.problems.{problem}.env")
    Env = getattr(mod, "Env")
    return Env(data_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Launch HeurAgenix hyper-heuristic on a problem.")
    parser.add_argument("-p", "--problem", type=str, required=True)
    parser.add_argument("-e", "--heuristic", type=str, required=True)  # llm_hh / random_hh / or a single heuristic
    parser.add_argument("-d", "--heuristic_dir", type=str, default="basic_heuristics")
    parser.add_argument("-t", "--test_data", type=str, default="")  # comma-separated file names, e.g. a280.tsp,case.json
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=1.0)
    parser.add_argument("-m", "--selection_frequency", type=int, default=5)
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1)
    parser.add_argument("-b", "--rollout_budget", type=int, default=0)
    parser.add_argument("-r", "--result_dir", type=str, default="result")
    parser.add_argument("-l", "--llm_config_file", type=str, default="azure_gpt_4o.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tool_calling", action="store_true")

    # backward compat (optional)
    parser.add_argument("-res", "--result_name", type=str, default=None)
    parser.add_argument("-exp", "--experiment_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # backward compat mapping
    if args.result_name and args.result_dir == "result":
        args.result_dir = args.result_name
    if args.experiment_name:
        args.heuristic = args.experiment_name

    random.seed(int(args.seed))

    # data root
    data_root = get_data_path()
    test_data_dir = data_root / args.problem / "test_data"
    if not test_data_dir.exists():
        # fallback: allow data_root/problem (search_file will still find)
        test_data_dir = data_root / args.problem

    # decide data_name_list (keep file names with extension)
    if args.test_data.strip():
        data_name_list = [x.strip() for x in args.test_data.split(",") if x.strip()]
    else:
        # list all files under test_data_dir
        data_name_list = []
        if test_data_dir.exists():
            for p in sorted(test_data_dir.iterdir()):
                if p.is_file():
                    data_name_list.append(p.name)

    if not data_name_list:
        raise RuntimeError(
            f"No test data found. test_data_dir={test_data_dir}, --test_data='{args.test_data}'"
        )

    # output base dir: <AMLT_OUTPUT_DIR>/output
    output_base_dir = get_output_dir()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # prepare heuristic pool if needed
    heuristic_pool = None
    if args.heuristic in ("llm_hh", "random_hh"):
        heuristic_pool = get_heuristic_names(args.problem, args.heuristic_dir)
        if not heuristic_pool:
            raise RuntimeError(f"Empty heuristic pool under {args.problem}/{args.heuristic_dir}")

    for data_name in data_name_list:
        test_stem = Path(data_name).stem
        out_dir = output_base_dir / args.problem / test_stem / args.result_dir / args.heuristic
        out_dir.mkdir(parents=True, exist_ok=True)

        env = _load_env(args.problem, data_name)
        env.reset(output_dir=str(out_dir))

        # IMPORTANT: make budget effective
        ps = int(getattr(env, "problem_size", 1) or 1)
        env.max_steps = max(1, int(float(args.iterations_scale_factor) * ps))
        env.construction_steps = 0

        if args.heuristic == "random_hh":
            runner = RandomHyperHeuristic(
                heuristic_pool=heuristic_pool,
                problem=args.problem,
                iterations_scale_factor=float(args.iterations_scale_factor),
                seed=int(args.seed),
            )
        elif args.heuristic == "llm_hh":
            llm_client = get_llm_client(
                args.llm_config_file,
                prompt_dir=str(REPO_ROOT / "src" / "problems" / "base" / "prompt"),
                output_dir=str(out_dir),
            )
            runner = LLMSelectionHyperHeuristic(
                llm_client=llm_client,
                heuristic_pool=heuristic_pool,
                problem=args.problem,
                tool_calling=bool(args.tool_calling),
                iterations_scale_factor=float(args.iterations_scale_factor),
                selection_frequency=int(args.selection_frequency),
                num_candidate_heuristics=int(args.num_candidate_heuristics),
                rollout_budget=int(args.rollout_budget),
                output_dir=str(out_dir),
                seed=int(args.seed),
            )
        else:
            runner = SingleHyperHeuristic(
                heuristic=args.heuristic,
                problem=args.problem,
                seed=int(args.seed),
            )

        runner.run(env)
        env.dump_result()


if __name__ == "__main__":
    main()
