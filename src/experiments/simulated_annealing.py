"""Required SA runs (3 per configuration) and run artifact serialization."""

import json
from collections.abc import Sequence
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.analysis import _spread
from src.experiments.genetic_algorithm import _serialize_state
from src.model import Package, State, Truck
from src.neighbors import random_initial_state
from src.objectives import OBJECTIVES, compactness, eta_weighted_value
from src.results import SearchResult
from src.visualization.genetic_algorithm import plot_state_comparison
from src.visualization.simulated_annealing import (
    plot_objective_comparison,
    plot_sa_acceptance,
    plot_sa_history,
    plot_sa_runs_summary,
)


def _serialize_result(result: SearchResult[State]) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "algorithm": result.algorithm,
        "objective_name": result.objective_name,
        "objective_value": result.objective_value,
        "objective_history": result.objective_history,
        "duration_seconds": result.duration_seconds,
        "seed": result.seed,
        "run_index": result.run_index,
        "parameters": dict(result.parameters),
        "metrics": {
            "iterations": metrics.iterations,
            "stuck_count": metrics.stuck_count,
            "stuck_patience": metrics.stuck_patience,
            "current_objective_history": metrics.current_objective_history,
            "exp_delta_e_over_temperature": metrics.exp_delta_e_over_temperature,
            "temperature_history": metrics.temperature_history,
        },
        "initial_state": _serialize_state(result.initial_state),
        "final_state": _serialize_state(result.final_state),
    }


def _summarize(results: Sequence[SearchResult[State]]) -> dict[str, Any]:
    finals = [result.final_state for result in results]
    return {
        "objective_name": results[0].objective_name,
        "run_count": len(results),
        "seeds": [result.seed for result in results],
        "parameters": dict(results[0].parameters),
        "final_objective_values": [result.objective_value for result in results],
        "final_objective_mean": mean(result.objective_value for result in results),
        "final_objective_stdev": _spread([result.objective_value for result in results]),
        "total_package_value_mean": mean(state.objective() for state in finals),
        "total_package_value_stdev": _spread([state.objective() for state in finals]),
        "eta_weighted_value_mean": mean(eta_weighted_value(state) for state in finals),
        "compactness_mean": mean(compactness(state) for state in finals),
        "packages_inside_mean": mean(
            sum(placement is not None for placement in state.placements.values())
            for state in finals
        ),
        "stuck_count_mean": mean(result.metrics.stuck_count for result in results),
        "duration_seconds_mean": mean(result.duration_seconds for result in results),
        "duration_seconds_stdev": _spread([result.duration_seconds for result in results]),
    }


def run_sa_experiments(
    packages: Sequence[Package],
    truck: Truck,
    *,
    seeds: Sequence[int],
    max_iterations: int,
    initial_temperature: float,
    cooling_rate: float,
    stuck_patience: int = 100,
    objective_name: str = "total_value",
    output_dir: str | Path = "output/sa",
) -> tuple[dict[str, Any], ...]:
    """Run SA once per seed (the spec asks for 3) and save plots plus JSON."""
    if len(seeds) < 1 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    if objective_name not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective_name!r}; choose from {sorted(OBJECTIVES)}")

    output_dir = Path(output_dir)
    packages = tuple(packages)
    records = []
    results = []
    for run_index, seed in enumerate(seeds, start=1):
        # Same derivation as the GA runner so both start from identical states.
        initial_seed = seed ^ 0x5EED
        initial_state = random_initial_state(packages, truck, Random(initial_seed))
        result = simulated_annealing(
            initial_state,
            seed=seed,
            objective_fn=OBJECTIVES[objective_name],
            objective_name=objective_name,
            max_iterations=max_iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            stuck_patience=stuck_patience,
            run_index=run_index,
        )
        results.append(result)

        run_dir = output_dir / f"run-{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        history_file = run_dir / "objective-history.png"
        acceptance_file = run_dir / "acceptance-probability.png"
        states_file = run_dir / "initial-final-states.png"
        plot_sa_history(result, history_file)
        plot_sa_acceptance(result, acceptance_file)
        plot_state_comparison(result.initial_state, result.final_state, states_file)

        record = {
            "run_index": run_index,
            "seed": seed,
            "initial_state_seed": initial_seed,
            "artifacts": {
                "run_directory": str(run_dir),
                "objective_history": str(history_file),
                "acceptance_probability": str(acceptance_file),
                "initial_final_states": str(states_file),
            },
            "result": _serialize_result(result),
        }
        (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)

    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(_summarize(results), indent=2), encoding="utf-8"
    )
    plot_sa_runs_summary(results, output_dir / "summary-runs.png")
    return tuple(records)


def compare_objectives(
    packages: Sequence[Package],
    truck: Truck,
    *,
    objective_names: Sequence[str],
    output_dir: str | Path = "output/sa",
    **sa_parameters: Any,
) -> tuple[dict[str, Any], ...]:
    """Run the SA experiment once per objective with identical seeds/parameters.

    Every final state is also scored on the required total_value so the impact
    of optimising an alternative objective can be compared directly.
    """
    output_dir = Path(output_dir)
    comparisons = []
    for objective_name in objective_names:
        run_sa_experiments(
            packages,
            truck,
            objective_name=objective_name,
            output_dir=output_dir / objective_name,
            **sa_parameters,
        )
        comparisons.append(
            json.loads((output_dir / objective_name / "summary.json").read_text(encoding="utf-8"))
        )
    (output_dir / "objective-comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    plot_objective_comparison(comparisons, output_dir / "objective-comparison.png")
    return tuple(comparisons)
