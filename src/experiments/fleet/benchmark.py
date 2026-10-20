"""Paired HC, SA, and GA benchmarks over the multi-truck fixture set."""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any, Callable

from src.algorithms.genetic_algorithm.fleet import (
    fleet_genetic_algorithm,
    random_initial_fleet_state,
)
from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.common.data import load_fleet_problem
from src.experiments.fleet.ga import _serialize_fleet_state
from src.model import FleetState, Truck
from src.objectives import OBJECTIVES
from src.validation import validate_fleet_state
from src.visualization.benchmark import plot_fleet_benchmark_summary
from src.visualization.genetic_algorithm import plot_fleet_state_comparison

HC_PARAMETERS = {
    "neighbors_per_iteration": 30,
    "max_iterations": 1000,
    "max_sideways": 100,
    "max_restarts": 10,
}
SA_PARAMETERS = {
    "max_iterations": 2000,
    "initial_temperature": 100.0,
    "cooling_rate": 0.995,
    "stuck_patience": 100,
}
GA_PARAMETERS = {"population_size": 20, "generations": 40, "mutation_rate": 0.05}


class _CountingObjective:
    def __init__(self, objective: Callable[[FleetState], int | float]) -> None:
        self.objective = objective
        self.evaluations = 0

    def __call__(self, state: FleetState) -> int | float:
        self.evaluations += 1
        return self.objective(state)


def discover_fleet_instances(
    fixtures_dir: str | Path,
) -> list[tuple[str, Path]]:
    fixtures_dir = Path(fixtures_dir)
    paths = sorted(fixtures_dir.glob("*.json")) + sorted(
        (fixtures_dir / "boundary").glob("*.json")
    )
    instances = []
    for path in paths:
        category = "boundary" if path.parent.name == "boundary" else path.stem
        if path.stem.startswith("tight-"):
            category = "tight"
        instances.append((category, path))
    return instances


def _method_specs():
    return [
        *((f"hc_{variant}", "hc", variant) for variant in VARIANTS),
        ("sa", "sa", None),
        ("ga", "ga", None),
    ]


def _run_method(
    method: str,
    algorithm: str,
    variant: str | None,
    initial: FleetState,
    seed: int,
    objective: _CountingObjective,
):
    if algorithm == "hc":
        return hill_climbing(
            initial,
            seed=seed,
            variant=variant,
            objective_fn=objective,
            objective_name="total_value",
            **HC_PARAMETERS,
        )
    if algorithm == "sa":
        return simulated_annealing(
            initial,
            seed=seed,
            objective_fn=objective,
            objective_name="total_value",
            **SA_PARAMETERS,
        )
    if algorithm == "ga":
        return fleet_genetic_algorithm(
            initial,
            seed=seed,
            objective_fn=objective,
            objective_name="total_value",
            **GA_PARAMETERS,
        )
    raise ValueError(f"unknown benchmark algorithm {algorithm!r} for {method}")


def _loaded_count(state: FleetState) -> int:
    return sum(
        placement is not None
        for truck_placements in state.placements
        for placement in truck_placements.values()
    )


def _expand_identical_trucks(
    trucks: tuple[Truck, ...], truck_count: int
) -> tuple[Truck, ...]:
    if type(truck_count) is not int or truck_count < 1:
        raise ValueError("truck_count must be a positive integer")
    if not trucks or any(truck != trucks[0] for truck in trucks[1:]):
        raise ValueError("source trucks must be identical to expand the fleet")
    return (trucks[0],) * truck_count


def summarize_fleet_benchmark(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                record["truck_count"],
                record["category"],
                record["instance"],
                record["method"],
            )
        ].append(record)

    summaries = []
    for (truck_count, category, instance, method), rows in groups.items():
        values = [row["final_value"] for row in rows]
        summaries.append(
            {
                "truck_count": truck_count,
                "category": category,
                "instance": instance,
                "method": method,
                "runs": len(rows),
                "initial_value_mean": mean(row["initial_value"] for row in rows),
                "initial_loaded_packages_mean": mean(
                    row["initial_loaded_packages"] for row in rows
                ),
                "final_value_mean": mean(values),
                "final_value_stdev": stdev(values) if len(values) > 1 else 0.0,
                "final_value_best_found": max(values),
                "improvement_mean": mean(
                    row["final_value"] - row["initial_value"] for row in rows
                ),
                "final_loaded_packages_mean": mean(
                    row["final_loaded_packages"] for row in rows
                ),
                "valid_rate": mean(float(row["valid"]) for row in rows),
                "evaluations_mean": mean(row["objective_evaluations"] for row in rows),
                "duration_seconds_mean": mean(row["duration_seconds"] for row in rows),
            }
        )
    return summaries


def summarize_initial_fleet_states(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize one shared initial state per fixture and seed, not per method."""
    groups: dict[tuple[int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[(record["truck_count"], record["instance"])].setdefault(
            record["seed"], record
        )

    summaries = []
    for (truck_count, instance), by_seed in groups.items():
        rows = list(by_seed.values())
        initial_states = [row["initial_state"] for row in rows]
        if any(len(state["trucks"]) != truck_count for state in initial_states):
            raise ValueError(f"initial state truck count does not match {truck_count} for {instance}")
        summaries.append(
            {
                "truck_count": truck_count,
                "instance": instance,
                "runs": len(rows),
                "initial_value_mean": mean(
                    state["total_package_value"] for state in initial_states
                ),
                "initial_value_stdev": stdev(
                    state["total_package_value"] for state in initial_states
                ) if len(rows) > 1 else 0.0,
                "loaded_packages_mean": mean(
                    sum(len(truck["packages"]) for truck in state["trucks"])
                    for state in initial_states
                ),
                "unloaded_packages_mean": mean(
                    len(state["unloaded_packages"]) for state in initial_states
                ),
                "valid_rate": mean(float(row["initial_valid"]) for row in rows),
                "per_truck": [
                    {
                        "truck_index": truck_index,
                        "package_count_mean": mean(
                            len(state["trucks"][truck_index]["packages"])
                            for state in initial_states
                        ),
                        "package_value_mean": mean(
                            state["trucks"][truck_index]["total_package_value"]
                            for state in initial_states
                        ),
                    }
                    for truck_index in range(truck_count)
                ],
            }
        )
    return summaries


def run_fleet_benchmark(
    fixtures_dir: str | Path = "tests/fixtures/fleet",
    *,
    seeds: tuple[int, ...] = (11,),
    truck_counts: tuple[int, ...] = (3, 4, 5),
    instance_names: tuple[str, ...] = ("base", "tight-space", "tight-capacity"),
    output_dir: str | Path = "output/fleet-count-analysis",
) -> tuple[dict[str, Any], ...]:
    """Run all HC variants, SA, and GA from paired seeded fleet states."""
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    if (
        not truck_counts
        or any(type(count) is not int or count < 1 for count in truck_counts)
        or len(set(truck_counts)) != len(truck_counts)
    ):
        raise ValueError("truck_counts must contain distinct positive integers")
    if (
        not instance_names
        or any(type(name) is not str or not name for name in instance_names)
        or len(set(instance_names)) != len(instance_names)
    ):
        raise ValueError("instance_names must contain distinct non-empty names")
    fixtures_dir, output_dir = Path(fixtures_dir), Path(output_dir)
    discovered_instances = discover_fleet_instances(fixtures_dir)
    if not discovered_instances:
        raise ValueError(f"no fleet fixtures found in {fixtures_dir}")
    available_names = {path.stem for _, path in discovered_instances}
    unknown_names = set(instance_names) - available_names
    if unknown_names:
        raise ValueError(f"unknown fleet fixtures {sorted(unknown_names)} in {fixtures_dir}")
    requested_names = set(instance_names)
    instances = [
        (category, path)
        for category, path in discovered_instances
        if path.stem in requested_names
    ]

    records: list[dict[str, Any]] = []
    methods = _method_specs()
    for category, path in instances:
        packages, source_trucks = load_fleet_problem(path)
        instance = path.stem
        for truck_count in truck_counts:
            trucks = _expand_identical_trucks(source_trucks, truck_count)
            for seed in seeds:
                initial_seed = seed ^ 0x5EED
                initial = random_initial_fleet_state(packages, trucks, Random(initial_seed))
                initial_record = _serialize_fleet_state(initial)
                initial_violations = validate_fleet_state(initial)
                for method, algorithm, variant in methods:
                    objective = _CountingObjective(OBJECTIVES["total_value"])
                    result = _run_method(method, algorithm, variant, initial, seed, objective)
                    final = result.final_state
                    violations = validate_fleet_state(final)
                    record = {
                        "truck_count": truck_count,
                        "category": category,
                        "instance": instance,
                        "method": method,
                        "algorithm": algorithm,
                        "variant": variant,
                        "seed": seed,
                        "initial_state_seed": initial_seed,
                        "initial_state": initial_record,
                        "initial_valid": not initial_violations,
                        "initial_violations": initial_violations,
                        "final_state": _serialize_fleet_state(final),
                        "initial_value": initial.objective(),
                        "final_value": final.objective(),
                        "initial_loaded_packages": _loaded_count(initial),
                        "final_loaded_packages": _loaded_count(final),
                        "valid": not violations,
                        "violations": violations,
                        "objective_evaluations": objective.evaluations,
                        "duration_seconds": result.duration_seconds,
                        "objective_history": list(result.objective_history),
                        "metrics": asdict(result.metrics),
                        "parameters": dict(result.parameters),
                    }
                    records.append(record)
                    if instance == "tight-space" and seed == seeds[0]:
                        plot_fleet_state_comparison(
                            initial,
                            final,
                            output_dir
                            / "representative-states"
                            / f"trucks-{truck_count}"
                            / f"{method}.png",
                        )
                    print(
                        f"{instance:20s} trucks={truck_count} seed={seed} {method:18s} "
                        f"value={record['final_value']} valid={record['valid']}",
                        flush=True,
                    )

    summaries = summarize_fleet_benchmark(records)
    initial_state_summaries = summarize_initial_fleet_states(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    # shortcut: aggregate JSON is written at completion; add per-run checkpoints if interrupted batches must resume.
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "hc": HC_PARAMETERS,
                    "sa": SA_PARAMETERS,
                    "ga": GA_PARAMETERS,
                    "objective": "total_value",
                    "seeds": list(seeds),
                    "truck_counts": list(truck_counts),
                    "initial_state_seed": "seed ^ 0x5EED",
                },
                "initial_state_summary": initial_state_summaries,
                "summary": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for truck_count in truck_counts:
        plot_fleet_benchmark_summary(
            [row for row in summaries if row["truck_count"] == truck_count],
            output_dir / f"summary-trucks-{truck_count}.png",
        )
    return tuple(records)
