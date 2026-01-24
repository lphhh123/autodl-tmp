import os
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.heuristic_only import HeuristicOnlyHyperHeuristic
from src.util.util import load_function

if os.environ.get("V54_ALLOW_RAW_HEURAGENIX", "0") != "1":
    raise RuntimeError(
        "v5.4 contract: Do NOT run HeurAgenix entrypoints directly (non-auditable). "
        "Use proj_ast2_ucf101_full/scripts/run_layout_heuragenix.py which emits v5.4 trace+seal. "
        "If you explicitly want non-v5.4 runs, set V54_ALLOW_RAW_HEURAGENIX=1."
    )


def _list_problems():
    problems_dir = Path(__file__).parent / "src" / "problems"
    out = []
    for d in problems_dir.iterdir():
        if d.is_dir() and d.name not in ["base", "__pycache__"]:
            out.append(d.name)
    return sorted(out)


def _resolve_test_data_dir(problem: str) -> Path:
    """
    v5.4 integration rule:
    - If AMLT_DATA_DIR is NOT set, data root must default to HeurAgenix repo_root/data,
      NOT the current working directory.
    - If AMLT_DATA_DIR is set and is relative, interpret it relative to repo_root.
    """
    repo_root = Path(__file__).resolve().parent
    env = os.environ.get("AMLT_DATA_DIR", None)
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        else:
            p = p.resolve()
        data_root = p
    else:
        data_root = (repo_root / "data").resolve()

    # data/{problem}/test_data
    test_dir = data_root / problem / "test_data"
    return test_dir


def _resolve_output_base(repo_root: Path) -> Path:
    """Resolve output base directory.

    Official: outputs go under:
      <output_base>/<problem>/<test_data>/<result_dir>/<engine>/...

    If AMLT_OUTPUT_DIR is set, use it as output base.
    """
    p = os.environ.get("AMLT_OUTPUT_DIR", "")
    if p:
        return Path(p).expanduser().resolve()
    return repo_root / "output"


def _resolve_roots(repo_root):
    import os
    import json
    from pathlib import Path

    amlt_data = (os.environ.get("AMLT_DATA_DIR") or "").strip()
    amlt_out = (os.environ.get("AMLT_OUTPUT_DIR") or "").strip()

    data_root = Path(amlt_data).expanduser() if amlt_data else (Path(repo_root) / "data")
    output_root = Path(amlt_out).expanduser() if amlt_out else (Path(repo_root) / "output")

    if data_root.is_absolute() is False:
        data_root = (Path(repo_root) / data_root).resolve()
    else:
        data_root = data_root.resolve()

    if output_root.is_absolute() is False:
        output_root = (Path(repo_root) / output_root).resolve()
    else:
        output_root = output_root.resolve()

    data_src = "AMLT_DATA_DIR" if amlt_data else "default_repo_root/data"
    out_src = "AMLT_OUTPUT_DIR" if amlt_out else "default_repo_root/output"

    # Make auditable: print + write a small meta file into output_root
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "data_root_source": data_src,
        "output_root_source": out_src,
    }
    try:
        (output_root / "heuragenix_roots.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"[HeurAgenix] data_root={data_root} (source={data_src})")
    print(f"[HeurAgenix] output_root={output_root} (source={out_src})")
    print(f"[HeurAgenix] effective data_root  = {data_root}")
    print(f"[HeurAgenix] effective output_root= {output_root}")

    return data_root, output_root


def _import_env(problem: str):
    mod = __import__(f"src.problems.{problem}.env", fromlist=["Env"])
    return getattr(mod, "Env")


def parse_arguments():
    problems = _list_problems()

    parser = argparse.ArgumentParser(description="Launch Hyper-Heuristic")
    parser.add_argument("-p", "--problem", type=str, required=True, choices=problems)
    parser.add_argument(
        "-e",
        "--heuristic",
        type=str,
        required=True,
        help=(
            "Heuristic / Hyper-heuristic engine. "
            "Use one of: llm_hh | random_hh | heuristic_only | or_solver | <heuristic_function_name> "
            "(or a .py file name under heuristic_dir)."
        ),
    )
    parser.add_argument("-d", "--heuristic_dir", type=str, default="basic_heuristics")
    parser.add_argument("-t", "--test_data", type=str, default=None,
                        help="Comma-separated test file names. Default: all in test_data dir.")
    parser.add_argument(
        "-l",
        "--llm_config_file",
        type=str,
        default=None,
        help="Path to the language model configuration file (REQUIRED by v5.4 contract).",
    )
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0)
    parser.add_argument("-m", "--selection_frequency", type=int, default=5)
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1)
    parser.add_argument("-b", "--rollout_budget", type=int, default=0)

    parser.add_argument("-r", "--result_dir", type=str, default="result")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional override for env.max_steps (exact budget). 0 means use iterations_scale_factor * problem_size.",
    )

    parser.add_argument("--llm_timeout_s", type=int, default=30)
    parser.add_argument("--max_llm_failures", type=int, default=2)
    parser.add_argument("--fallback_on_llm_failure", type=str, default="random_hh",
                        choices=["random_hh", "stop"])

    parser.add_argument("-res", "--result_name", type=str, default=None)
    parser.add_argument("-exp", "--experiment_name", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = Path(__file__).resolve().parent
    llm_config_effective = None
    if args.llm_config_file:
        llm_config_effective = str(Path(args.llm_config_file).expanduser().resolve())
    data_root, output_base = _resolve_roots(repo_root)
    test_data_dir = data_root / args.problem / "test_data"

    if args.test_data is None or args.test_data.strip() == "":
        data_name_list = sorted([p.name for p in test_data_dir.iterdir() if p.is_file()])
    else:
        data_name_list = [x.strip() for x in args.test_data.split(",") if x.strip()]

    heur_dir = repo_root / "src" / "problems" / args.problem / "heuristics" / args.heuristic_dir
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
        if int(args.max_steps) > 0:
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

        if args.heuristic == "heuristic_only":
            runner = HeuristicOnlyHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                heuristic_dir=str(heur_dir),
                iterations_scale_factor=float(args.iterations_scale_factor),
                selection_frequency=int(args.selection_frequency),
                output_dir=str(out_dir),
                seed=seed_val,
            )
        elif args.heuristic == "random_hh":
            runner = RandomHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                iterations_scale_factor=float(args.iterations_scale_factor),
                heuristic_dir=str(heur_dir),
                selection_frequency=int(args.selection_frequency),
                seed=seed_val,
            )
        elif args.heuristic == "llm_hh":
            if not args.llm_config_file:
                raise ValueError(
                    "[v5.4] --llm_config_file is required when using llm_hh. "
                    "Refusing to fall back to implicit defaults."
                )
            llm_meta = {
                "llm_config_file": llm_config_effective,
                "llm_config_file_exists": bool(
                    llm_config_effective and Path(llm_config_effective).is_file()
                ),
                "llm_config_file_source": "cli",
            }
            (out_dir / "llm_config_meta.json").write_text(
                json.dumps(llm_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            runner = LLMSelectionHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                heuristic_dir=str(heur_dir),
                llm_config_file=llm_config_effective,
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
        elif args.heuristic == "or_solver":
            print(
                "[WARN] or_solver is declared in docs but not implemented in this repo snapshot. "
                "Falling back to heuristic_only."
            )
            runner = HeuristicOnlyHyperHeuristic(
                heuristic_pool=heuristic_pool_files,
                problem=args.problem,
                heuristic_dir=str(heur_dir),
                iterations_scale_factor=float(args.iterations_scale_factor),
                selection_frequency=int(args.selection_frequency),
                output_dir=str(out_dir),
                seed=seed_val,
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
