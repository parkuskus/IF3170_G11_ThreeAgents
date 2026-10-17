import argparse
from pathlib import Path
from typing import Sequence

from src.algorithms.hill_climbing import VARIANTS
from src.experiments.data import load_problem
from src.experiments.hill_climbing import run_hc_experiments


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
    args = parser.parse_args(argv)

    packages, truck = load_problem(args.input)
    records = run_hc_experiments(
        packages,
        truck,
        seeds=args.seeds,
        variants=args.variants,
        output_dir=args.output,
        neighbors_per_iteration=args.neighbors_per_iteration,
        max_iterations=args.max_iterations,
        max_sideways=args.max_sideways,
        max_restarts=args.max_restarts,
    )
    print(f"Completed {len(records)} runs. Results: {args.output / 'results.json'}")
    print(f"Aggregate summary: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
