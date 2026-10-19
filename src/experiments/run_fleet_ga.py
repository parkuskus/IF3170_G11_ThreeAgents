"""CLI for a single Genetic Algorithm run over a configurable truck fleet."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.data import load_fleet_problem
from src.experiments.fleet_genetic_algorithm import run_fleet_ga


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize package loading across a truck fleet.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/fleet-ga"))
    parser.add_argument("--truck-count", type=int, help="replicate legacy truck config N times")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    packages, trucks = load_fleet_problem(args.input, truck_count=args.truck_count)
    record = run_fleet_ga(
        packages,
        trucks,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        output_dir=args.output,
    )
    print(
        f"Completed fleet GA: value {record['result']['objective_value']}; "
        f"results: {args.output / 'result.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
