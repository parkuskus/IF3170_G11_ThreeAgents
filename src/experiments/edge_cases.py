"""Run comparable GA edge cases against the same package list."""

import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.model import Package, State, Truck
from src.neighbors import random_initial_state
from src.visualization.genetic_algorithm import plot_edge_case_comparison


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def run_edge_case_comparison(
    packages: tuple[Package, ...],
    base_truck: Truck,
    *,
    seeds: tuple[int, int, int],
    population_size: int,
    generations: int,
    output_dir: str | Path = "output/ga-edge-cases",
    mutation_rate: float = 0.05,
) -> tuple[dict[str, Any], ...]:
    """Run 3 seeds for baseline, tight-capacity, compact, and zero-capacity cases."""
    if len(set(seeds)) != 3 or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must contain three distinct integers")
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be positive")
    if type(generations) is not int or generations < 0:
        raise ValueError("generations must be non-negative")

    compact_dimensions = tuple(max(1, dimension // 2) for dimension in base_truck.dimensions)
    cases = (
        ("baseline", base_truck),
        (
            "tight-capacity",
            Truck(base_truck.dimensions, base_truck.max_capacity // 2),
        ),
        ("compact-truck", Truck(compact_dimensions, base_truck.max_capacity)),
        ("zero-capacity", Truck(base_truck.dimensions, 0)),
    )
    output_dir = Path(output_dir)
    records = []
    summaries = []

    for case_name, truck in cases:
        case_runs = []
        for run_index, seed in enumerate(seeds, start=1):
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            result = genetic_algorithm(
                initial,
                seed=seed,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                objective_name="total_value",
                run_index=run_index,
            )
            run = {
                "case": case_name,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "objective_value": result.objective_value,
                "duration_seconds": result.duration_seconds,
                "best_history": result.metrics.best_objective_history,
                "average_history": result.metrics.average_objective_history,
            }
            records.append(run)
            case_runs.append(run)

        best_histories = [run["best_history"] for run in case_runs]
        average_histories = [run["average_history"] for run in case_runs]
        summaries.append(
            {
                "case": case_name,
                "truck_dimensions": truck.dimensions,
                "max_capacity": truck.max_capacity,
                "run_count": len(case_runs),
                "final_objective_mean": mean(run["objective_value"] for run in case_runs),
                "final_objective_stdev": _spread(
                    [run["objective_value"] for run in case_runs]
                ),
                "duration_seconds_mean": mean(
                    run["duration_seconds"] for run in case_runs
                ),
                "best_history_mean": [
                    mean(history[i] for history in best_histories)
                    for i in range(generations + 1)
                ],
                "best_history_stdev": [
                    _spread([history[i] for history in best_histories])
                    for i in range(generations + 1)
                ],
                "average_history_mean": [
                    mean(history[i] for history in average_histories)
                    for i in range(generations + 1)
                ],
                "average_history_stdev": [
                    _spread([history[i] for history in average_histories])
                    for i in range(generations + 1)
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    plot_edge_case_comparison(summaries, output_dir / "objective-vs-generations.png")
    return tuple(summaries)
