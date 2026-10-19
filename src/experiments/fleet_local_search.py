"""HC and SA experiment runners for multi-truck fleet states."""

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Any

from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.genetic_algorithm.fleet import random_initial_fleet_state
from src.experiments.fleet_genetic_algorithm import _serialize_fleet_state
from src.experiments.objective_comparison import aggregate_objective_runs, write_objective_comparison
from src.model import FleetState, Package, Truck
from src.objectives import OBJECTIVES
from src.results import SearchResult
from src.visualization.genetic_algorithm import plot_fleet_state_comparison
from src.visualization.hill_climbing import plot_hc_history, plot_hc_summary
from src.visualization.simulated_annealing import (
    plot_objective_comparison,
    plot_sa_acceptance,
    plot_sa_history,
    plot_sa_runs_summary,
)


def _serialize_result(result: SearchResult[FleetState]) -> dict[str, Any]:
    return {
        "algorithm": result.algorithm,
        "objective_name": result.objective_name,
        "objective_value": result.objective_value,
        "objective_history": result.objective_history,
        "duration_seconds": result.duration_seconds,
        "seed": result.seed,
        "run_index": result.run_index,
        "parameters": dict(result.parameters),
        "metrics": asdict(result.metrics),
        "initial_state": _serialize_fleet_state(result.initial_state),
        "final_state": _serialize_fleet_state(result.final_state),
    }


def _save_run(
    result: SearchResult[FleetState], run_dir: Path, algorithm: str
) -> dict[str, Any]:
    history_path = run_dir / "objective-history.png"
    states_path = run_dir / "initial-final-states.png"
    if algorithm == "hill_climbing":
        plot_hc_history(result, history_path)
    else:
        plot_sa_history(result, history_path)
        plot_sa_acceptance(result, run_dir / "acceptance-probability.png")
    plot_fleet_state_comparison(result.initial_state, result.final_state, states_path)
    artifacts = {
        "run_directory": str(run_dir),
        "objective_history": str(history_path),
        "initial_final_states": str(states_path),
    }
    if algorithm == "simulated_annealing":
        artifacts["acceptance_probability"] = str(run_dir / "acceptance-probability.png")
    record = {
        "seed": result.seed,
        "run_index": result.run_index,
        "objective_name": result.objective_name,
        "variant": result.metrics.variant if algorithm == "hill_climbing" else None,
        "result": _serialize_result(result),
        "artifacts": artifacts,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _validate_experiment(seeds: tuple[int, ...], objective_names: tuple[str, ...]) -> None:
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    unknown = [name for name in objective_names if name not in OBJECTIVES]
    if unknown:
        raise ValueError(f"unknown objectives: {unknown}; choose from {sorted(OBJECTIVES)}")
    if not objective_names:
        raise ValueError("at least one objective is required")


def run_fleet_hc(
    packages: tuple[Package, ...],
    trucks: tuple[Truck, ...],
    *,
    seeds: tuple[int, ...] = (41,),
    objective_names: tuple[str, ...] = ("total_value",),
    output_dir: str | Path = "output/fleet-hc",
    variants: tuple[str, ...] = ("steepest",),
    neighbors_per_iteration: int = 30,
    max_iterations: int = 1000,
    max_sideways: int = 100,
    max_restarts: int = 10,
) -> tuple[dict[str, Any], ...]:
    seeds, objective_names, variants = tuple(seeds), tuple(objective_names), tuple(variants)
    _validate_experiment(seeds, objective_names)
    unknown_variants = [variant for variant in variants if variant not in VARIANTS]
    if unknown_variants:
        raise ValueError(f"unknown Hill-climbing variants: {unknown_variants}")
    if not variants:
        raise ValueError("at least one Hill-climbing variant is required")

    output_dir = Path(output_dir)
    records: list[dict[str, Any]] = []
    comparisons = []
    summaries = []
    for objective_name in objective_names:
        objective_records = []
        objective_summaries = []
        for variant in variants:
            variant_records = []
            for run_index, seed in enumerate(seeds, start=1):
                initial = random_initial_fleet_state(
                    tuple(packages), tuple(trucks), Random(seed ^ 0x5EED)
                )
                result = hill_climbing(
                    initial,
                    seed=seed,
                    objective_fn=OBJECTIVES[objective_name],
                    objective_name=objective_name,
                    variant=variant,
                    neighbors_per_iteration=neighbors_per_iteration,
                    max_iterations=max_iterations,
                    max_sideways=max_sideways,
                    max_restarts=max_restarts,
                    run_index=run_index,
                )
                record = _save_run(
                    result,
                    output_dir / objective_name / variant / f"run-{run_index}",
                    "hill_climbing",
                )
                variant_records.append(record)
                objective_records.append(record)
                records.append(record)
            variant_summary = {
                "objective_name": objective_name,
                **summarize_hc_runs(variant_records)[0],
            }
            objective_summaries.append(variant_summary)
            summaries.append(variant_summary)
        (output_dir / objective_name / "results.json").write_text(
            json.dumps(objective_records, indent=2), encoding="utf-8"
        )
        (output_dir / objective_name / "summary.json").write_text(
            json.dumps(objective_summaries, indent=2), encoding="utf-8"
        )
        plot_hc_summary(objective_summaries, output_dir / objective_name / "summary.png")
        comparisons.append(aggregate_objective_runs(objective_name, objective_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if len(records) == 1:
        (output_dir / "result.json").write_text(json.dumps(records[0], indent=2), encoding="utf-8")
    if len(objective_names) > 1:
        write_objective_comparison(
            comparisons,
            output_dir,
            "Hill-climbing fleet total package value by objective function (mean ± 1 SD)",
        )
    return tuple(records)


def summarize_hc_runs(records: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = [record["result"] for record in records]
    objectives = [result["objective_value"] for result in results]
    durations = [result["duration_seconds"] for result in results]
    iterations = [result["metrics"]["iterations"] for result in results]
    return [{
        "variant": results[0]["metrics"]["variant"],
        "runs": len(results),
        "objective_mean": mean(objectives),
        "objective_stdev": pstdev(objectives),
        "objective_best": max(objectives),
        "duration_mean": mean(durations),
        "iterations_mean": mean(iterations),
    }]


def run_fleet_sa(
    packages: tuple[Package, ...],
    trucks: tuple[Truck, ...],
    *,
    seeds: tuple[int, ...] = (41,),
    objective_names: tuple[str, ...] = ("total_value",),
    output_dir: str | Path = "output/fleet-sa",
    max_iterations: int = 2000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    stuck_patience: int = 100,
) -> tuple[dict[str, Any], ...]:
    seeds, objective_names = tuple(seeds), tuple(objective_names)
    _validate_experiment(seeds, objective_names)
    output_dir = Path(output_dir)
    records: list[dict[str, Any]] = []
    comparisons = []
    summaries = []
    for objective_name in objective_names:
        objective_records = []
        results = []
        for run_index, seed in enumerate(seeds, start=1):
            initial = random_initial_fleet_state(
                tuple(packages), tuple(trucks), Random(seed ^ 0x5EED)
            )
            result = simulated_annealing(
                initial,
                seed=seed,
                objective_fn=OBJECTIVES[objective_name],
                objective_name=objective_name,
                max_iterations=max_iterations,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                stuck_patience=stuck_patience,
                run_index=run_index,
            )
            record = _save_run(
                result, output_dir / objective_name / f"run-{run_index}", "simulated_annealing"
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
            "stuck_count_mean": mean(result.metrics.stuck_count for result in results),
            "duration_seconds_mean": mean(result.duration_seconds for result in results),
        }
        summaries.append(summary)
        (output_dir / objective_name / "results.json").write_text(
            json.dumps(objective_records, indent=2), encoding="utf-8"
        )
        (output_dir / objective_name / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        plot_sa_runs_summary(results, output_dir / objective_name / "summary-runs.png")
        comparisons.append(aggregate_objective_runs(objective_name, objective_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if len(records) == 1:
        (output_dir / "result.json").write_text(json.dumps(records[0], indent=2), encoding="utf-8")
    if len(objective_names) > 1:
        plot_objective_comparison(
            comparisons,
            output_dir / "objective-comparison.png",
            title="SA fleet total package value by objective function (mean ± 1 SD)",
        )
        (output_dir / "objective-comparison.json").write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8"
        )
    return tuple(records)
