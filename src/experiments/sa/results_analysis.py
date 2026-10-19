"""Reproducible SA fixture benchmark and one-factor parameter sweeps."""

import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.common.benchmark import discover_instances
from src.experiments.common.data import load_problem
from src.experiments.sa.experiments import _serialize_result
from src.model import State
from src.neighbors import random_initial_state
from src.results import SearchResult
from src.validation import validate_state
from src.visualization.genetic_algorithm import plot_state_comparison
from src.visualization.simulated_annealing import plot_sa_acceptance, plot_sa_history


SEEDS = (11, 12, 13, 14, 15)
BASELINE = {
    "max_iterations": 3000,
    "initial_temperature": 100.0,
    "cooling_rate": 0.995,
    "stuck_patience": 100,
}
TEMPERATURE_VALUES = (10.0, 100.0, 1000.0)
COOLING_RATES = (0.99, 0.995, 0.999)
ITERATION_VALUES = (1000, 3000, 6000)


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _worse_moves_accepted(current_history: tuple[float, ...]) -> int:
    return sum(after < before for before, after in zip(current_history, current_history[1:]))


def _iteration_of_best(objective_history: tuple[float, ...]) -> int:
    return objective_history.index(objective_history[-1])


def summarize_sa_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate SA runs by experiment and the parameter that was varied."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        factor = record["factor"]
        value = str(record["factor_value"])
        groups[(record["experiment"], record["instance"], factor, value)].append(record)

    summaries = []
    for (experiment, instance, factor, value), runs in sorted(groups.items()):
        objectives = [run["result"]["objective_value"] for run in runs]
        summaries.append(
            {
                "experiment": experiment,
                "factor": factor,
                "factor_value": (
                    "all" if factor == "baseline"
                    else int(value) if factor == "max_iterations"
                    else float(value)
                ),
                "instance": instance,
                "run_count": len(runs),
                "seeds": [run["seed"] for run in runs],
                "valid_rate": mean(1.0 if run["valid"] else 0.0 for run in runs),
                "initial_value_mean": mean(run["initial_value"] for run in runs),
                "final_value_mean": mean(objectives),
                "final_value_stdev": _spread(objectives),
                "final_value_best": max(objectives),
                "loaded_packages_mean": mean(run["loaded_packages"] for run in runs),
                "worse_accepted_mean": mean(run["worse_accepted"] for run in runs),
                "best_iteration_mean": mean(run["best_iteration"] for run in runs),
                "stuck_count_mean": mean(run["stuck_count"] for run in runs),
                "final_temperature": runs[0]["final_temperature"],
                "duration_seconds_mean": mean(run["duration_seconds"] for run in runs),
                "evaluations_mean": mean(run["evaluations"] for run in runs),
            }
        )
    return summaries


def _configurations(instances):
    for category, name, path in instances:
        yield "baseline", category, name, path, "baseline", "all", dict(BASELINE)
    base = next(item for item in instances if item[1] == "base")
    for value in TEMPERATURE_VALUES:
        yield "temperature-sweep", *base, "initial_temperature", value, {
            **BASELINE,
            "initial_temperature": value,
        }
    for value in COOLING_RATES:
        yield "cooling-sweep", *base, "cooling_rate", value, {
            **BASELINE,
            "cooling_rate": value,
        }
    for value in ITERATION_VALUES:
        yield "iteration-sweep", *base, "max_iterations", value, {
            **BASELINE,
            "max_iterations": value,
        }


def _run_directory(
    output_dir: Path, experiment: str, instance: str, factor: str,
    factor_value: int | float | str, seed: int,
) -> Path:
    return output_dir / experiment / instance / f"{factor}-{factor_value}" / f"seed-{seed}"


def _plot_fixture_summary(summaries, destination: Path) -> None:
    rows = [row for row in summaries if row["experiment"] == "baseline"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(rows))
    ax.bar(x, [row["final_value_mean"] for row in rows],
           yerr=[row["final_value_stdev"] for row in rows], capsize=3)
    ax.scatter(x, [row["initial_value_mean"] for row in rows], color="black",
               marker="_", s=300, zorder=3, label="Initial state (mean)")
    ax.set_xticks(list(x), [row["instance"] for row in rows], rotation=35, ha="right")
    ax.set(title="SA baseline by fixture (mean ± 1 SD, five seeds)",
           ylabel="Best-found total package value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _plot_parameter_sweep(summaries, experiment: str, destination: Path) -> None:
    rows = sorted(
        (row for row in summaries if row["experiment"] == experiment),
        key=lambda row: row["factor_value"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = [str(row["factor_value"]) for row in rows]
    axes[0].errorbar(x, [row["final_value_mean"] for row in rows],
                     yerr=[row["final_value_stdev"] for row in rows], marker="o", capsize=4)
    axes[0].set(title="Best-found value (mean ± 1 SD)", xlabel=rows[0]["factor"], ylabel="Package value")
    axes[1].plot(x, [row["worse_accepted_mean"] for row in rows], marker="o")
    axes[1].set(title="Worse moves accepted (mean)", xlabel=rows[0]["factor"], ylabel="Moves")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"SA {experiment.replace('-', ' ')} on base (five seeds)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_sa_analysis(
    fixtures_dir: str | Path = "tests/fixtures",
    output_dir: str | Path = "output/sa-analysis",
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[dict[str, Any], ...]:
    """Run five-seed fixture baselines and paired one-factor sweeps on base."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must contain at least two distinct integers")
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    records = []
    representative_plotted = set()
    instances = discover_instances(Path(fixtures_dir))
    for experiment, category, instance, path, factor, factor_value, parameters in _configurations(instances):
        packages, truck = load_problem(path)
        for run_index, seed in enumerate(seeds, 1):
            # Same derivation as the GA and benchmark runners: identical initial states.
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            result: SearchResult[State] = simulated_annealing(
                initial, seed=seed, run_index=run_index,
                objective_name="total_value", **parameters,
            )
            serialized = _serialize_result(result)
            violations = validate_state(result.final_state)
            run_dir = _run_directory(
                output_dir, experiment, instance, factor, factor_value, seed
            )
            metrics = result.metrics
            record = {
                "experiment": experiment,
                "category": category,
                "instance": instance,
                "factor": factor,
                "factor_value": factor_value,
                "parameters": parameters,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "initial_value": result.initial_state.objective(),
                "loaded_packages": sum(p is not None for p in result.final_state.placements.values()),
                "valid": not violations,
                "violations": violations,
                "worse_accepted": _worse_moves_accepted(metrics.current_objective_history),
                "best_iteration": _iteration_of_best(result.objective_history),
                "stuck_count": metrics.stuck_count,
                "final_temperature": (
                    parameters["initial_temperature"]
                    * parameters["cooling_rate"] ** parameters["max_iterations"]
                ),
                "evaluations": parameters["max_iterations"] + 1,
                "duration_seconds": result.duration_seconds,
                "artifact_path": str(run_dir / "result.json"),
                "result": serialized,
            }
            records.append(record)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            if experiment == "baseline" and instance not in representative_plotted:
                plot_sa_history(result, figure_dir / instance / "objective-history.png")
                plot_sa_acceptance(result, figure_dir / instance / "acceptance-probability.png")
                plot_state_comparison(
                    result.initial_state, result.final_state,
                    figure_dir / instance / "initial-final-states.png",
                )
                representative_plotted.add(instance)
            print(f"{experiment:17s} {instance:20s} seed={seed} value={result.objective_value}", flush=True)

    summaries = summarize_sa_analysis(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"seeds": list(seeds), "baseline": BASELINE,
                    "sweeps": {"initial_temperature": list(TEMPERATURE_VALUES),
                               "cooling_rate": list(COOLING_RATES),
                               "max_iterations": list(ITERATION_VALUES)},
                    "summary": summaries}, indent=2), encoding="utf-8"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_fixture_summary(summaries, figure_dir / "fixture-summary.png")
    for experiment in ("temperature-sweep", "cooling-sweep", "iteration-sweep"):
        _plot_parameter_sweep(summaries, experiment, figure_dir / f"{experiment}.png")
    return tuple(records)


if __name__ == "__main__":
    runs = run_sa_analysis()
    print(f"Completed {len(runs)} SA runs under output/sa-analysis")
