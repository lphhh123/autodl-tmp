# HeurAgenix/launch_hyper_heuristic.py
import argparse
from pathlib import Path

from src.pipeline.hyper_heuristic import launch_heuristic, launch_heuristic_selector
from src.util.data_path import get_data_path
from src.util.output_dir import get_output_dir


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


def iter_test_files(test_data_dir: Path, test_data_arg):
    if test_data_arg:
        names = [x.strip() for x in test_data_arg.split(",") if x.strip()]
        return [test_data_dir / n for n in names]
    return sorted([p for p in test_data_dir.iterdir() if p.is_file()])


def main():
    args = parse_args()

    test_data_dir = get_data_path() / args.problem / "test_data"
    output_dir = get_output_dir()
    data_name_list = iter_test_files(test_data_dir, args.test_data)

    if args.heuristic in {"llm_hh", "random_hh"}:
        launch_heuristic_selector(
            args.problem,
            test_data_dir,
            data_name_list,
            output_dir,
            args.heuristic,
            args.heuristic_dir,
            args.llm_config_file,
            args.iterations_scale_factor,
            args.selection_frequency,
            args.num_candidate_heuristics,
            args.rollout_budget,
            result_dir=args.result_dir,
        )
    else:
        launch_heuristic(
            args.problem,
            test_data_dir,
            data_name_list,
            output_dir,
            args.heuristic,
            args.iterations_scale_factor,
            result_dir=args.result_dir,
        )


if __name__ == "__main__":
    main()
