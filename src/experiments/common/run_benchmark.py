"""CLI: run every algorithm on every fixture instance with paired seeds."""

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.common.benchmark import run_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark HC, SA, and GA on all fixture instances.")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 12, 13, 14, 15))
    args = parser.parse_args(argv)
    records = run_benchmark(args.fixtures, seeds=tuple(args.seeds), output_dir=args.output)
    print(f"Completed {len(records)} runs. Results: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
