"""Single-run fleet GA runner and JSON artifact serialization."""

import json
from pathlib import Path
from random import Random
from typing import Any

from src.algorithms.genetic_algorithm.fleet import fleet_genetic_algorithm, random_initial_fleet_state
from src.model import FleetState, Package, Truck
from src.results import SearchResult
from src.visualization.genetic_algorithm import plot_fleet_state_comparison, plot_ga_history


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
    initial_state = random_initial_fleet_state(packages, trucks, Random(seed ^ 0x5EED))
    result = fleet_genetic_algorithm(
        initial_state,
        seed=seed,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )
    output_dir = Path(output_dir)
    history_path = output_dir / "objective-history.png"
    states_path = output_dir / "initial-final-states.png"
    plot_ga_history(result, history_path)
    plot_fleet_state_comparison(result.initial_state, result.final_state, states_path)
    record = {
        "result": _serialize_result(result),
        "artifacts": {
            "objective_history": str(history_path),
            "initial_final_states": str(states_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
