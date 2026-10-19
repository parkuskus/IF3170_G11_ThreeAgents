"""Command-line entry point for the required GA parameter sweeps."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.data import load_problem
from src.experiments.genetic_algorithm import compare_ga_objectives, run_ga_experiments
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the two required GA experiments.")
    parser.add_argument("--input", required=True, type=Path, help="problem JSON file")
    parser.add_argument("--output", type=Path, default=Path("output/ga"))
    parser.add_argument("--generation-values", required=True, nargs=3, type=int)
    parser.add_argument("--population-values", required=True, nargs=3, type=int)
    parser.add_argument("--fixed-population-size", required=True, type=int)
    parser.add_argument("--fixed-generations", required=True, type=int)
    parser.add_argument("--seeds", required=True, nargs=3, type=int)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=sorted(OBJECTIVES),
        default=["total_value"],
        help="objective(s) to optimise; several also write an objective comparison",
    )
    args = parser.parse_args(argv)

    packages, truck = load_problem(args.input)
    ga_parameters = dict(
        generation_values=args.generation_values,
        population_values=args.population_values,
        fixed_population_size=args.fixed_population_size,
        fixed_generations=args.fixed_generations,
        seeds=args.seeds,
        mutation_rate=args.mutation_rate,
    )
    if args.objectives == ["total_value"]:
        records = run_ga_experiments(packages, truck, output_dir=args.output, **ga_parameters)
        print(f"Completed {len(records)} runs. Results: {args.output / 'results.json'}")
        print(f"Aggregate summaries: {args.output / 'summary.json'}")
        return 0
    comparisons = compare_ga_objectives(
        packages, truck, objective_names=args.objectives, output_dir=args.output, **ga_parameters
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
