"""CLI for one Simulated Annealing run over a configurable truck fleet."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.data import load_fleet_problem
from src.experiments.fleet_local_search import run_fleet_sa
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Simulated Annealing over a truck fleet.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/fleet-sa"))
    parser.add_argument("--truck-count", type=int, help="replicate legacy truck config N times")
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", type=int, help="run once with this seed (default: 41)")
    seeds_group.add_argument("--seeds", nargs=3, type=int, help="run three distinct seeds")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--stuck-patience", type=int, default=100)
    parser.add_argument(
        "--objectives", nargs="+", choices=sorted(OBJECTIVES), default=["total_value"]
    )
    args = parser.parse_args(argv)

    packages, trucks = load_fleet_problem(args.input, truck_count=args.truck_count)
    seeds = args.seeds or (args.seed if args.seed is not None else 41,)
    records = run_fleet_sa(
        packages,
        trucks,
        seeds=seeds,
        objective_names=tuple(args.objectives),
        output_dir=args.output,
        max_iterations=args.max_iterations,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        stuck_patience=args.stuck_patience,
    )
    print(
        f"Completed {len(records)} fleet SA runs; results: {args.output / 'results.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
