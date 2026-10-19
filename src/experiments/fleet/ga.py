"""Single-run fleet GA runner and JSON artifact serialization."""

import json
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Any

from src.algorithms.genetic_algorithm.fleet import fleet_genetic_algorithm, random_initial_fleet_state
from src.model import FleetState, Package, Truck
from src.objectives import OBJECTIVES
from src.results import SearchResult
from src.experiments.common.objective_comparison import (
    aggregate_objective_runs,
    write_objective_comparison,
)
from src.visualization.genetic_algorithm import (
    plot_fleet_state_comparison,
    plot_ga_history,
)


def _serialize_fleet_state(state: FleetState) -> dict[str, Any]:
    trucks = []
    assigned = set()
    for index, (truck, placements) in enumerate(zip(state.trucks, state.placements)):
        loaded = []
        for package in state.packages:
            placement = placements[package.id]
            if placement is None:
                continue
            assigned.add(package.id)
            loaded.append(
                {
                    "id": package.id,
                    "dimensions": package.dimensions,
                    "value": package.value,
                    "weight": package.weight,
                    "fragile": package.fragile,
                    "eta": package.eta,
                    "coordinate": placement.coordinate,
                    "orientation": placement.orientation,
                    "oriented_dimensions": placement.oriented_dimensions,
                }
            )
        trucks.append(
            {
                "truck_index": index,
                "dimensions": truck.dimensions,
                "max_capacity": truck.max_capacity,
                "total_package_value": sum(item["value"] for item in loaded),
                "packages": loaded,
            }
        )
    return {
        "truck_count": len(state.trucks),
        "total_package_value": state.objective(),
        "trucks": trucks,
        "unloaded_packages": [
            package.id for package in state.packages if package.id not in assigned
        ],
    }


def _serialize_result(result: SearchResult[FleetState]) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "algorithm": result.algorithm,
        "objective_name": result.objective_name,
        "objective_value": result.objective_value,
        "objective_history": result.objective_history,
        "duration_seconds": result.duration_seconds,
        "seed": result.seed,
        "parameters": dict(result.parameters),
        "metrics": {
            "generations": metrics.generations,
            "population_size": metrics.population_size,
            "best_objective_history": metrics.best_objective_history,
            "average_objective_history": metrics.average_objective_history,
        },
        "initial_state": _serialize_fleet_state(result.initial_state),
        "final_state": _serialize_fleet_state(result.final_state),
    }


def run_fleet_ga(
    packages: tuple[Package, ...],
    trucks: tuple[Truck, ...],
    *,
    seed: int,
    population_size: int,
    generations: int,
    mutation_rate: float = 0.05,
    output_dir: str | Path = "output/fleet-ga",
) -> dict[str, Any]:
    return run_fleet_ga_experiments(
        packages,
        trucks,
        seeds=(seed,),
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        output_dir=output_dir,
    )[0]


def run_fleet_ga_experiments(
    packages: tuple[Package, ...],
    trucks: tuple[Truck, ...],
    *,
    seeds: tuple[int, ...],
    population_size: int,
    generations: int,
    mutation_rate: float = 0.05,
    objective_names: tuple[str, ...] = ("total_value",),
    output_dir: str | Path = "output/fleet-ga",
) -> tuple[dict[str, Any], ...]:
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    if not objective_names:
        raise ValueError("at least one objective is required")
    unknown = [name for name in objective_names if name not in OBJECTIVES]
    if unknown:
        raise ValueError(f"unknown objectives: {unknown}; choose from {sorted(OBJECTIVES)}")

    output_dir = Path(output_dir)
    records: list[dict[str, Any]] = []
    summaries = []
    comparisons = []
    for objective_name in objective_names:
        objective_records = []
        results = []
        for run_index, seed in enumerate(seeds, start=1):
            initial = random_initial_fleet_state(packages, trucks, Random(seed ^ 0x5EED))
            result = fleet_genetic_algorithm(
                initial,
                seed=seed,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                objective_fn=OBJECTIVES[objective_name],
                objective_name=objective_name,
            )
            run_dir = (
                output_dir
                if len(seeds) == len(objective_names) == 1
                else output_dir / objective_name / f"run-{run_index}"
            )
            history_path = run_dir / "objective-history.png"
            states_path = run_dir / "initial-final-states.png"
            plot_ga_history(result, history_path)
            plot_fleet_state_comparison(result.initial_state, result.final_state, states_path)
            record = {
                "seed": seed,
                "run_index": run_index,
                "objective_name": objective_name,
                "result": _serialize_result(result),
                "artifacts": {
                    "run_directory": str(run_dir),
                    "objective_history": str(history_path),
                    "initial_final_states": str(states_path),
                },
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            objective_records.append(record)
            records.append(record)
            results.append(result)

        final_values = [result.final_state.objective() for result in results]
        summary = {
            "objective_name": objective_name,
            "run_count": len(results),
            "seeds": list(seeds),
            "final_objective_values": [result.objective_value for result in results],
            "final_objective_mean": mean(result.objective_value for result in results),
            "final_objective_stdev": pstdev(result.objective_value for result in results),
            "total_package_value_mean": mean(final_values),
            "total_package_value_stdev": pstdev(final_values),
            "duration_seconds_mean": mean(result.duration_seconds for result in results),
        }
        summaries.append(summary)
        objective_dir = output_dir / objective_name
        objective_dir.mkdir(parents=True, exist_ok=True)
        (objective_dir / "results.json").write_text(
            json.dumps(objective_records, indent=2), encoding="utf-8"
        )
        (objective_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        comparisons.append(aggregate_objective_runs(objective_name, objective_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if len(records) == 1:
        (output_dir / "result.json").write_text(
            json.dumps(records[0], indent=2), encoding="utf-8"
        )
    if len(objective_names) > 1:
        write_objective_comparison(
            comparisons,
            output_dir,
            "GA fleet total package value by objective function (mean ± 1 SD)",
        )
    return tuple(records)
