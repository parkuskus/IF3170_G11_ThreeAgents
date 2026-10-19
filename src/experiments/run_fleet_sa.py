"""Compatibility entry point for the fleet SA CLI."""

from src.experiments.fleet.run_sa import main

if __name__ == "__main__":
    raise SystemExit(main())
