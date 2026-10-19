"""CLI for one Hill-climbing run over a configurable truck fleet."""

import argparse
from pathlib import Path
from typing import Sequence

from src.algorithms.hill_climbing import VARIANTS
from src.experiments.common.data import load_fleet_problem
from src.experiments.fleet.local_search import run_fleet_hc
from src.objectives import OBJECTIVES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hill-climbing over a truck fleet.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/fleet-hc"))
    parser.add_argument("--truck-count", type=int, help="replicate legacy truck config N times")
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", type=int, help="run once with this seed (default: 41)")
    seeds_group.add_argument("--seeds", nargs=3, type=int, help="run three distinct seeds")
    parser.add_argument("--variant", choices=VARIANTS, default="steepest")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--neighbors-per-iteration", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-sideways", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument(
        "--objectives", nargs="+", choices=sorted(OBJECTIVES), default=["total_value"]
    )
    args = parser.parse_args(argv)

    packages, trucks = load_fleet_problem(args.input, truck_count=args.truck_count)
    seeds = args.seeds or (args.seed if args.seed is not None else 41,)
    records = run_fleet_hc(
        packages,
        trucks,
        seeds=seeds,
        objective_names=tuple(args.objectives),
        output_dir=args.output,
        variants=tuple(args.variants or (args.variant,)),
        neighbors_per_iteration=args.neighbors_per_iteration,
        max_iterations=args.max_iterations,
        max_sideways=args.max_sideways,
        max_restarts=args.max_restarts,
    )
    print(
        f"Completed {len(records)} fleet HC runs; results: {args.output / 'results.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
