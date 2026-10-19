import argparse
from pathlib import Path
from typing import Sequence

from src.algorithms.hill_climbing import VARIANTS
from src.experiments.common.data import load_problem
from src.experiments.hc.experiments import compare_hc_objectives, run_hc_experiments
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hill-climbing variants, 3 runs each.")
    parser.add_argument("--input", required=True, type=Path, help="problem JSON file")
    parser.add_argument("--output", type=Path, default=Path("output/hill-climbing"))
    parser.add_argument("--seeds", required=True, nargs=3, type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--neighbors-per-iteration", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-sideways", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=sorted(OBJECTIVES),
        default=["total_value"],
        help="objective(s) to optimise; several also write an objective comparison",
    )
    args = parser.parse_args(argv)

    packages, truck = load_problem(args.input)
    hc_parameters = dict(
        seeds=args.seeds,
        variants=args.variants,
        neighbors_per_iteration=args.neighbors_per_iteration,
        max_iterations=args.max_iterations,
        max_sideways=args.max_sideways,
        max_restarts=args.max_restarts,
    )
    if args.objectives == ["total_value"]:
        records = run_hc_experiments(packages, truck, output_dir=args.output, **hc_parameters)
        print(f"Completed {len(records)} runs. Results: {args.output / 'results.json'}")
        print(f"Aggregate summary: {args.output / 'summary.json'}")
        return 0
    comparisons = compare_hc_objectives(
        packages, truck, objective_names=args.objectives, output_dir=args.output, **hc_parameters
    )
    for comparison in comparisons:
        print(
            f"{comparison['objective_name']}: total value "
            f"{comparison['total_package_value_mean']:g} +/- {comparison['total_package_value_stdev']:g}"
        )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
