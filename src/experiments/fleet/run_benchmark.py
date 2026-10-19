"""CLI for paired multi-truck algorithm benchmarks."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.fleet.benchmark import run_fleet_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare HC, SA, and GA on paired multi-truck fixtures."
    )
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/fleet"))
    parser.add_argument("--output", type=Path, default=Path("output/fleet-analysis"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 12, 13, 14, 15))
    args = parser.parse_args(argv)
    records = run_fleet_benchmark(
        args.fixtures, seeds=tuple(args.seeds), output_dir=args.output
    )
    print(f"Completed {len(records)} fleet runs; results: {args.output / 'runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
