"""Shared helpers for running HC and GA under the alternative objectives."""

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from src.contracts import ObjectiveFunction
from src.experiments.common.analysis import _spread
from src.model import State
from src.objectives import OBJECTIVES
from src.visualization.simulated_annealing import plot_objective_comparison


def resolve_objective_fn(
    objective_fn: ObjectiveFunction[State] | None, objective_name: str | None
) -> ObjectiveFunction[State] | None:
    """Look up a registered objective by name unless an explicit function is given."""
    if objective_name is not None and objective_name not in OBJECTIVES and objective_fn is None:
        raise ValueError(f"unknown objective {objective_name!r}; choose from {sorted(OBJECTIVES)}")
    if objective_fn is None and objective_name is not None:
        return OBJECTIVES[objective_name]
    return objective_fn


def aggregate_objective_runs(
    objective_name: str, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Score every final state of one objective on the required total value."""
    finals = [record["result"]["final_state"] for record in records]
    totals = [final["total_package_value"] for final in finals]
    return {
        "objective_name": objective_name,
        "run_count": len(records),
        "total_package_value_mean": mean(totals),
        "total_package_value_stdev": _spread(totals),
        "final_objective_mean": mean(record["result"]["objective_value"] for record in records),
        "duration_seconds_mean": mean(record["result"]["duration_seconds"] for record in records),
    }


def write_objective_comparison(
    comparisons: Sequence[dict[str, Any]], output_dir: Path, title: str
) -> None:
    (output_dir / "objective-comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    plot_objective_comparison(comparisons, output_dir / "objective-comparison.png", title=title)
