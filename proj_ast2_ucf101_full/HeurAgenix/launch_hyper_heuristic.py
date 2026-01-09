# HeurAgenix/launch_hyper_heuristic.py
import argparse
import glob
import importlib
import math
import os
from pathlib import Path

from src.pipeline.hyper_heuristics import LLMSelectionHyperHeuristic, RandomHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import load_heuristic_functions


def parse_args():
    problems_root = Path(__file__).resolve().parent / "src" / "problems"
    problem_pool = [p.name for p in problems_root.iterdir() if p.is_dir() and p.name != "base"]

    ap = argparse.ArgumentParser("launch_hyper_heuristic (README compatible)")
    ap.add_argument("-p", "--problem", choices=problem_pool, required=True)
    ap.add_argument("-e", "--heuristic", required=True, help="heuristic function name OR llm_hh OR random_hh")
    ap.add_argument("-l", "--llm_config_file", default=None)
    ap.add_argument("-d", "--heuristic_dir", default="basic_heuristics")
    ap.add_argument("-t", "--test_data", default=None, help="comma-separated filenames under data/{problem}/test_data")
    ap.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0)
    ap.add_argument("-m", "--selection_frequency", type=int, default=5)
    ap.add_argument("-c", "--num_candidate_heuristics", type=int, default=1)
    ap.add_argument("-b", "--rollout_budget", type=int, default=0)
    ap.add_argument("-r", "--result_dir", default="result")

    # SPEC extensions (safe defaults)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm_timeout_s", type=int, default=30)
    ap.add_argument("--max_llm_failures", type=int, default=2)
    ap.add_argument("--fallback_mode", type=str, default="random_hh")
    return ap.parse_args()


def iter_test_files(problem, test_data_arg):
    test_root = Path("data") / problem / "test_data"
    if test_data_arg:
        names = [x.strip() for x in test_data_arg.split(",") if x.strip()]
        return [test_root / n for n in names]
    return sorted([Path(p) for p in glob.glob(str(test_root / "*")) if Path(p).is_file()])


def build_env(problem, data_name, seed):
    module = importlib.import_module(f"src.problems.{problem}.env")
    Env = getattr(module, "Env")
    env = Env(data_name=str(data_name), seed=seed) if "seed" in Env.__init__.__code__.co_varnames else Env(str(data_name))
    return env


def main():
    args = parse_args()

    out_base = Path("output") / args.problem / "test_data" / args.result_dir
    out_base.mkdir(parents=True, exist_ok=True)

    for case_path in iter_test_files(args.problem, args.test_data):
        env = build_env(args.problem, case_path, args.seed)

        # output dir per case
        case_ref = getattr(env, "data_ref_name", case_path.stem)
        out_dir = out_base / case_ref / f"seed{args.seed}_{args.heuristic}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # reset env to target output dir
        env.reset(output_dir=str(out_dir))

        # compute steps
        problem_size = getattr(env, "problem_size", getattr(env, "construction_steps", 1))
        total_steps = max(1, int(math.ceil(args.iterations_scale_factor * float(problem_size))))

        configs = {
            "problem": args.problem,
            "heuristic_dir": args.heuristic_dir,
            "llm_config_file": args.llm_config_file,
            "selection_frequency": args.selection_frequency,
            "num_candidate_heuristics": args.num_candidate_heuristics,
            "rollout_budget": args.rollout_budget,
            "iterations_scale_factor": args.iterations_scale_factor,
            "max_steps": total_steps,
            "llm_timeout_s": args.llm_timeout_s,
            "max_llm_failures": args.max_llm_failures,
            "fallback_mode": args.fallback_mode,
        }

        # load heuristics
        heur_funcs = load_heuristic_functions(args.problem, args.heuristic_dir)

        if args.heuristic == "llm_hh":
            llm_client = None
            llm_error = None
            if args.llm_config_file:
                try:
                    llm_client = get_llm_client(args.llm_config_file)
                except Exception as exc:  # noqa: BLE001
                    llm_client = None
                    llm_error = f"{type(exc).__name__}: {exc}"
            hh = LLMSelectionHyperHeuristic(configs=configs, llm_client=llm_client, llm_error=llm_error)
            hh.heuristic_functions = heur_funcs
            ok = hh.run(env)
        elif args.heuristic == "random_hh":
            hh = RandomHyperHeuristic(configs=configs)
            hh.heuristic_functions = heur_funcs
            ok = hh.run(env)
        else:
            # direct single heuristic by name
            if args.heuristic not in heur_funcs:
                raise ValueError(f"Unknown heuristic {args.heuristic}. Available: {list(heur_funcs.keys())[:20]}...")
            # run repeatedly
            for step in range(total_steps):
                op, meta = heur_funcs[args.heuristic](env.get_problem_state(), algorithm_data={"env": env, "rng": env.rng})
                if op is None:
                    break
                env.run_operator(op, heuristic_name=args.heuristic)
            ok = True

        # dump results (recordings.jsonl + best_solution.json)
        env.dump_result()


if __name__ == "__main__":
    main()
