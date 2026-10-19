"""Compatibility entry point for the SA CLI."""

from src.experiments.sa.run import main

if __name__ == "__main__":
    raise SystemExit(main())
