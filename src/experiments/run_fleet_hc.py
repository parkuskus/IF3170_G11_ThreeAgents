"""Compatibility entry point for the fleet HC CLI."""

from src.experiments.fleet.run_hc import main

if __name__ == "__main__":
    raise SystemExit(main())
