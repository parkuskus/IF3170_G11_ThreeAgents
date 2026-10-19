"""CLI for a single Genetic Algorithm run over a configurable truck fleet."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.common.data import load_fleet_problem
from src.experiments.fleet.ga import run_fleet_ga_experiments
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize package loading across a truck fleet.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/fleet-ga"))
    parser.add_argument("--truck-count", type=int, help="replicate legacy truck config N times")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", type=int, help="run once with this seed (default: 41)")
    seeds_group.add_argument("--seeds", nargs=3, type=int, help="run three distinct seeds")
    parser.add_argument(
        "--objectives", nargs="+", choices=sorted(OBJECTIVES), default=["total_value"]
    )
    args = parser.parse_args(argv)
    packages, trucks = load_fleet_problem(args.input, truck_count=args.truck_count)
    seeds = args.seeds or (args.seed if args.seed is not None else 41,)
    records = run_fleet_ga_experiments(
        packages,
        trucks,
        seeds=seeds,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        objective_names=tuple(args.objectives),
        output_dir=args.output,
    )
    print(
        f"Completed {len(records)} fleet GA runs; results: {args.output / 'results.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
