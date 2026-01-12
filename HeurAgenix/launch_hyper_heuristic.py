import argparse
import os
import random
from pathlib import Path

import numpy as np


def _get_problem_pool():
    problems_dir = Path("src") / "problems"
    pool = []
    if problems_dir.exists():
        for d in problems_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "base":
                pool.append(d.name)
    return sorted(pool)


def _resolve_heuristic_dir(problem: str, heuristic_dir: str) -> Path:
    p = Path(heuristic_dir)
    if p.is_absolute():
        return p
    return Path("src") / "problems" / problem / "heuristics" / heuristic_dir


def _list_heuristics(heur_dir: Path):
    if not heur_dir.exists():
        return []
    hs = []
    for f in sorted(heur_dir.glob("*.py")):
        if f.name.startswith("__"):
            continue
        hs.append(f.stem)
    return hs


def main():
    parser = argparse.ArgumentParser(
        description="Launch Hyper Heuristic (patched for wrapper/spec)"
    )

    parser.add_argument("-p", "--problem", type=str, required=True, choices=_get_problem_pool())
    parser.add_argument(
        "-e",
        "--heuristic",
        type=str,
        required=True,
        help="heuristic name OR llm_hh OR random_hh",
    )
    parser.add_argument(
        "-d",
        "--heuristic_dir",
        type=str,
        default="basic_heuristics",
        help="subdir under src/problems/{problem}/heuristics, or absolute path",
    )

    parser.add_argument(
        "-t",
        "--test_data",
        type=str,
        default=None,
        help=(
            "comma-split list of test filenames (e.g., case_seed1.json). "
            "If None, run all in test_data/"
        ),
    )
    parser.add_argument("-l", "--llm_config_file", type=str, default=None)

    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0)
    parser.add_argument("-m", "--selection_frequency", type=int, default=5)
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1)
    parser.add_argument("-b", "--rollout_budget", type=int, default=0)

    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        default="result",
        help="result folder name (default: result)",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    problem = args.problem
    engine = args.heuristic

    amlt_out = os.getenv("AMLT_OUTPUT_DIR", None)
    base_output_dir = (Path(amlt_out) / "output") if amlt_out else Path("output")

    amlt_data = os.getenv("AMLT_DATA_DIR", None)
    data_root = Path(amlt_data) if amlt_data else Path("data")
    test_data_dir = data_root / problem / "test_data"

    if args.test_data is None:
        data_name_list = [p.name for p in sorted(test_data_dir.glob("*.json"))]
    else:
        data_name_list = [x.strip() for x in args.test_data.split(",") if x.strip()]

    if not data_name_list:
        raise RuntimeError(
            f"No test cases found. test_data_dir={test_data_dir}, test_data={args.test_data}"
        )

    mod = __import__(f"src.problems.{problem}.env", fromlist=["Env"])
    Env = getattr(mod, "Env")

    heur_dir = _resolve_heuristic_dir(problem, args.heuristic_dir)
    heuristic_pool = _list_heuristics(heur_dir)

    for data_name in data_name_list:
        case_stem = Path(data_name).stem
        out_dir = os.path.join(
            base_output_dir, problem, case_stem, args.result_dir, args.heuristic
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        env = Env(data_name=data_name)
        env.reset(output_dir=str(out_dir))

        if getattr(env, "construction_steps", None) is None:
            s_value = env.instance_data.get("S", None)
            if s_value is None:
                chiplets = env.instance_data.get("chiplets", {}).get("S", None)
                s_value = chiplets
            env.construction_steps = int(s_value) if s_value else 1

        usage_path = Path(out_dir) / "llm_usage.jsonl"
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        if not usage_path.exists():
            usage_path.write_text("", encoding="utf-8")

        if engine == "random_hh":
            from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic

            iterations_scale_factor = float(args.iterations_scale_factor)
            if args.max_steps is not None:
                iterations_scale_factor = max(
                    1.0,
                    float(args.max_steps) / max(1, env.construction_steps),
                )
            runner = RandomHyperHeuristic(
                heuristic_pool=heuristic_pool,
                problem=problem,
                iterations_scale_factor=iterations_scale_factor,
                heuristic_dir=str(heur_dir),
            )
            runner.run(env)

        elif engine == "llm_hh":
            from src.pipeline.hyper_heuristics.llm_selection import (
                LLMSelectionHyperHeuristic,
            )
            from src.util.llm_client.get_llm_client import get_llm_client

            llm_ok = True
            llm_client = None
            try:
                prompt_dir = str(Path("src") / "problems" / problem / "prompt")
                llm_client = get_llm_client(
                    args.llm_config_file,
                    prompt_dir=prompt_dir,
                    output_dir=str(out_dir),
                )
            except Exception as e:
                llm_ok = False
                with open(usage_path, "a", encoding="utf-8") as f:
                    f.write(
                        f'{{"ok": false, "reason": "llm_client_init_failed", "error": "{str(e)}"}}\n'
                    )

            if llm_ok:
                iterations_scale_factor = float(args.iterations_scale_factor)
                if args.max_steps is not None:
                    iterations_scale_factor = max(
                        1.0,
                        float(args.max_steps) / max(1, env.construction_steps),
                    )
                runner = LLMSelectionHyperHeuristic(
                    heuristic_pool=heuristic_pool,
                    llm_client=llm_client,
                    problem=problem,
                    iterations_scale_factor=iterations_scale_factor,
                    selection_frequency=int(args.selection_frequency),
                    num_candidate_heuristics=int(args.num_candidate_heuristics),
                    rollout_budget=int(args.rollout_budget),
                )
                runner.run(env)
            else:
                from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic

                runner = RandomHyperHeuristic(
                    heuristic_pool=heuristic_pool,
                    problem=problem,
                    iterations_scale_factor=float(args.iterations_scale_factor),
                    heuristic_dir=str(heur_dir),
                )
                runner.run(env)

        else:
            from src.util.util import load_function

            fn = load_function(engine, problem=problem)
            if args.max_steps is not None:
                max_steps = int(args.max_steps)
            else:
                max_steps = int(env.construction_steps * float(args.iterations_scale_factor))
            while env.current_steps < max_steps and env.continue_run:
                env.run_heuristic(fn)

        env.dump_result()
    print(f"[HeurAgenix] run finished seed={args.seed}, results saved to {base_output_dir}")


if __name__ == "__main__":
    main()
