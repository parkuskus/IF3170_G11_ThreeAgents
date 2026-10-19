"""Required GA parameter sweeps and run artifact serialization."""

import json
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.experiments.common.analysis import summarize_ga_runs
from src.contracts import ObjectiveFunction
from src.experiments.common.objective_comparison import (
    aggregate_objective_runs,
    resolve_objective_fn,
    write_objective_comparison,
)
from src.model import Package, State, Truck
from src.neighbors import random_initial_state
from src.results import SearchResult
from src.visualization.genetic_algorithm import (
    plot_ga_history,
    plot_ga_sweep_summary,
    plot_state_comparison,
)


def _validate_values(values: Sequence[int], name: str, *, allow_zero: bool) -> tuple[int, int, int]:
    if len(values) != 3 or any(
        type(value) is not int or value < (0 if allow_zero else 1) for value in values
    ):
        requirement = "three distinct non-negative integers" if allow_zero else "three distinct positive integers"
        raise ValueError(f"{name} must contain {requirement}")
    if len(set(values)) != 3:
        raise ValueError(f"{name} values must be distinct")
    if tuple(sorted(values)) != tuple(values):
        raise ValueError(f"{name} values must be in increasing order")
    return tuple(values)


def _serialize_state(state: State) -> dict[str, Any]:
    return {
        "truck": {
            "dimensions": state.truck.dimensions,
            "max_capacity": state.truck.max_capacity,
        },
        "total_package_value": state.objective(),
        "packages": [
            {
                "id": package.id,
                "dimensions": package.dimensions,
                "value": package.value,
                "weight": package.weight,
                "fragile": package.fragile,
                "eta": package.eta,
                "placement": (
                    None
                    if (placement := state.placements[package.id]) is None
                    else {
                        "coordinate": placement.coordinate,
                        "orientation": placement.orientation,
                        "oriented_dimensions": placement.oriented_dimensions,
                    }
                ),
            }
            for package in state.packages
        ],
    }


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
            "generations": metrics.generations,
            "population_size": metrics.population_size,
            "best_objective_history": metrics.best_objective_history,
            "average_objective_history": metrics.average_objective_history,
        },
        "initial_state": _serialize_state(result.initial_state),
        "final_state": _serialize_state(result.final_state),
    }


def _run_one(
    packages: tuple[Package, ...],
    truck: Truck,
    *,
    experiment: str,
    population_size: int,
    generations: int,
    run_index: int,
    seed: int,
    mutation_rate: float,
    output_dir: Path,
    objective_fn: ObjectiveFunction[State] | None,
    objective_name: str | None,
) -> dict[str, Any]:
    initial_seed = seed ^ 0x5EED
    initial_state = random_initial_state(packages, truck, Random(initial_seed))
    result = genetic_algorithm(
        initial_state,
        seed=seed,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        objective_fn=objective_fn,
        objective_name=objective_name,
        run_index=run_index,
    )

    run_dir = output_dir / experiment / f"population-{population_size}" / f"generations-{generations}" / f"run-{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_file = run_dir / "objective-history.png"
    states_file = run_dir / "initial-final-states.png"
    plot_ga_history(result, history_file)
    plot_state_comparison(result.initial_state, result.final_state, states_file)

    record = {
        "experiment": experiment,
        "population_size": population_size,
        "generations": generations,
        "run_index": run_index,
        "seed": seed,
        "initial_state_seed": initial_seed,
        "artifacts": {
            "run_directory": str(run_dir),
            "objective_history": str(history_file),
            "initial_final_states": str(states_file),
        },
        "result": _serialize_result(result),
    }
    (run_dir / "result.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def run_ga_experiments(
    packages: Sequence[Package],
    truck: Truck,
    *,
    generation_values: Sequence[int],
    population_values: Sequence[int],
    fixed_population_size: int,
    fixed_generations: int,
    seeds: Sequence[int],
    output_dir: str | Path = "output/ga",
    mutation_rate: float = 0.05,
    objective_fn: ObjectiveFunction[State] | None = None,
    objective_name: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run both required GA sweeps: 3 values × 3 seeds for each parameter."""
    generation_values = _validate_values(
        generation_values, "generation_values", allow_zero=True
    )
    population_values = _validate_values(
        population_values, "population_values", allow_zero=False
    )
    if type(fixed_population_size) is not int or fixed_population_size < 1:
        raise ValueError("fixed_population_size must be a positive integer")
    if type(fixed_generations) is not int or fixed_generations < 0:
        raise ValueError("fixed_generations must be a non-negative integer")
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ValueError("seeds must contain three distinct integers")

    objective_fn = resolve_objective_fn(objective_fn, objective_name)
    output_dir = Path(output_dir)
    packages = tuple(packages)
    records = []
    for experiment, configs in (
        (
            "generation-sweep",
            ((fixed_population_size, generations) for generations in generation_values),
        ),
        (
            "population-sweep",
            ((population_size, fixed_generations) for population_size in population_values),
        ),
    ):
        for population_size, generations in configs:
            for run_index, seed in enumerate(seeds, start=1):
                records.append(
                    _run_one(
                        packages,
                        truck,
                        experiment=experiment,
                        population_size=population_size,
                        generations=generations,
                        run_index=run_index,
                        seed=seed,
                        mutation_rate=mutation_rate,
                        output_dir=output_dir,
                        objective_fn=objective_fn,
                        objective_name=objective_name,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    summaries = summarize_ga_runs(tuple(records))
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    for experiment in ("generation-sweep", "population-sweep"):
        plot_ga_sweep_summary(
            summaries, experiment, output_dir / f"summary-{experiment}.png"
        )
    return tuple(records)


def compare_ga_objectives(
    packages: Sequence[Package],
    truck: Truck,
    *,
    objective_names: Sequence[str],
    output_dir: str | Path = "output/ga",
    **ga_parameters: Any,
) -> tuple[dict[str, Any], ...]:
    """Run both GA sweeps once per objective with identical seeds/parameters."""
    output_dir = Path(output_dir)
    comparisons = []
    for objective_name in objective_names:
        records = run_ga_experiments(
            packages,
            truck,
            objective_name=objective_name,
            output_dir=output_dir / objective_name,
            **ga_parameters,
        )
        comparisons.append(aggregate_objective_runs(objective_name, records))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_objective_comparison(
        comparisons,
        output_dir,
        "GA final total package value by objective function (mean ± 1 SD)",
    )
    return tuple(comparisons)
