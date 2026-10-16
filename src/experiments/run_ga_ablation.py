"""CLI for the four-variant GA operator ablation study."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.ablation import run_ga_ablation
from src.experiments.data import load_problem


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare GA baseline with three operator ablations.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/ga-ablation"))
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43, 44, 45))
    args = parser.parse_args(argv)
    packages, truck = load_problem(args.input)
    records = run_ga_ablation(
        packages,
        truck,
        seeds=tuple(args.seeds),
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        output_dir=args.output,
    )
    print(f"Completed {len(records)} runs across 4 variants. Results: {args.output / 'runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
