"""Compatibility entry point for the fleet GA CLI."""

from src.experiments.fleet.run_ga import main

if __name__ == "__main__":
    raise SystemExit(main())
