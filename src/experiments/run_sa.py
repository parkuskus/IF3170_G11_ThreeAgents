"""Command-line entry point for the required Simulated Annealing runs."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.data import load_problem
from src.experiments.simulated_annealing import compare_objectives
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Simulated Annealing 3 times per objective.")
    parser.add_argument("--input", required=True, type=Path, help="problem JSON file")
    parser.add_argument("--output", type=Path, default=Path("output/sa"))
    parser.add_argument("--seeds", required=True, nargs=3, type=int)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--stuck-patience", type=int, default=100)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=sorted(OBJECTIVES),
        default=["total_value"],
        help="objective(s) to optimise; several also write an objective comparison",
    )
    args = parser.parse_args(argv)

    packages, truck = load_problem(args.input)
    comparisons = compare_objectives(
        packages,
        truck,
        objective_names=args.objectives,
        output_dir=args.output,
        seeds=args.seeds,
        max_iterations=args.max_iterations,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        stuck_patience=args.stuck_patience,
    )
    for comparison in comparisons:
        print(
            f"{comparison['objective_name']}: total value "
            f"{comparison['total_package_value_mean']:g} +/- {comparison['total_package_value_stdev']:g}, "
            f"stuck {comparison['stuck_count_mean']:g}x, "
            f"{comparison['duration_seconds_mean']:.2f}s per run"
        )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
