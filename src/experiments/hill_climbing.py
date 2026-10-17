import json
from collections.abc import Sequence
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Any

from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.contracts import ObjectiveFunction
from src.experiments.genetic_algorithm import _serialize_state
from src.model import Package, State, Truck
from src.neighbors import random_initial_state
from src.results import SearchResult
from src.visualization.genetic_algorithm import plot_state_comparison
from src.visualization.hill_climbing import plot_hc_history, plot_hc_summary


def _serialize_result(result: SearchResult[State]) -> dict[str, Any]:
    m = result.metrics
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
            "variant": m.variant,
            "iterations": m.iterations,
            "sideways_moves": m.sideways_moves,
            "maximum_sideways_moves": m.maximum_sideways_moves,
            "restarts": m.restarts,
            "maximum_restarts": m.maximum_restarts,
            "iterations_per_restart": m.iterations_per_restart,
        },
        "initial_state": _serialize_state(result.initial_state),
        "final_state": _serialize_state(result.final_state),
    }


def summarize_hc_runs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for variant in dict.fromkeys(r["variant"] for r in records):
        rows = [r["result"] for r in records if r["variant"] == variant]
        objectives = [r["objective_value"] for r in rows]
        durations = [r["duration_seconds"] for r in rows]
        iterations = [r["metrics"]["iterations"] for r in rows]
        summaries.append(
            {
                "variant": variant,
                "runs": len(rows),
                "objective_mean": mean(objectives),
                "objective_stdev": pstdev(objectives),
                "objective_best": max(objectives),
                "duration_mean": mean(durations),
                "iterations_mean": mean(iterations),
            }
        )
    return summaries


def run_hc_experiments(
    packages: Sequence[Package],
    truck: Truck,
    *,
    seeds: Sequence[int],
    variants: Sequence[str] = VARIANTS,
    output_dir: str | Path = "output/hill-climbing",
    neighbors_per_iteration: int = 30,
    max_iterations: int = 1000,
    max_sideways: int = 100,
    max_restarts: int = 10,
    objective_fn: ObjectiveFunction[State] | None = None,
    objective_name: str | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(seeds) != 3 or any(type(s) is not int for s in seeds) or len(set(seeds)) != 3:
        raise ValueError("seeds harus berisi tiga bilangan bulat yang berbeda")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"varian tidak dikenal: {unknown}")

    output_dir, packages = Path(output_dir), tuple(packages)
    records = []
    for variant in variants:
        for run_index, seed in enumerate(seeds, start=1):
            initial_seed = seed ^ 0x5EED
            initial_state = random_initial_state(packages, truck, Random(initial_seed))
            result = hill_climbing(
                initial_state,
                seed=seed,
                variant=variant,
                neighbors_per_iteration=neighbors_per_iteration,
                max_iterations=max_iterations,
                max_sideways=max_sideways,
                max_restarts=max_restarts,
                objective_fn=objective_fn,
                objective_name=objective_name,
                run_index=run_index,
            )
            run_dir = output_dir / variant / f"run-{run_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            plot_hc_history(result, run_dir / "objective-history.png")
            plot_state_comparison(
                result.initial_state, result.final_state, run_dir / "initial-final-states.png"
            )
            record = {
                "variant": variant,
                "run_index": run_index,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "artifacts": {
                    "run_directory": str(run_dir),
                    "objective_history": str(run_dir / "objective-history.png"),
                    "initial_final_states": str(run_dir / "initial-final-states.png"),
                },
                "result": _serialize_result(result),
            }
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summaries = summarize_hc_runs(records)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_hc_summary(summaries, output_dir / "summary.png")
    return tuple(records)
