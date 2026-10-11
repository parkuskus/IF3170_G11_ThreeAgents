"""Command-line entry point for the required GA parameter sweeps."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.data import load_problem
from src.experiments.genetic_algorithm import run_ga_experiments


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
    args = parser.parse_args(argv)

    packages, truck = load_problem(args.input)
    records = run_ga_experiments(
        packages,
        truck,
        generation_values=args.generation_values,
        population_values=args.population_values,
        fixed_population_size=args.fixed_population_size,
        fixed_generations=args.fixed_generations,
        seeds=args.seeds,
        output_dir=args.output,
        mutation_rate=args.mutation_rate,
    )
    print(f"Completed {len(records)} runs. Results: {args.output / 'results.json'}")
    print(f"Aggregate summaries: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
