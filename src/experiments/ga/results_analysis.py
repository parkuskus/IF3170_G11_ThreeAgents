"""Reproducible GA fixture benchmark and one-factor parameter sweeps."""

import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.experiments.common.benchmark import discover_instances
from src.experiments.common.data import load_problem
from src.experiments.ga.experiments import _serialize_result
from src.model import State
from src.neighbors import random_initial_state
from src.results import SearchResult
from src.validation import validate_state
from src.visualization.genetic_algorithm import plot_ga_history, plot_state_comparison


SEEDS = (11, 12, 13, 14, 15)
BASELINE = {"population_size": 20, "generations": 40, "mutation_rate": 0.05}
POPULATION_VALUES = (10, 20, 40)
GENERATION_VALUES = (20, 40, 80)
MUTATION_RATES = (0.0, 0.05, 0.2)


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_ga_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate GA runs by experiment and the parameter that was varied."""
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
                    else float(value) if factor == "mutation_rate"
                    else int(value)
                ),
                "instance": instance,
                "run_count": len(runs),
                "seeds": [run["seed"] for run in runs],
                "valid_rate": mean(1.0 if run["valid"] else 0.0 for run in runs),
                "final_value_mean": mean(objectives),
                "final_value_stdev": _spread(objectives),
                "final_value_best": max(objectives),
                "loaded_packages_mean": mean(run["loaded_packages"] for run in runs),
                "duration_seconds_mean": mean(run["duration_seconds"] for run in runs),
                "evaluations_mean": mean(run["evaluations"] for run in runs),
            }
        )
    return summaries


def _configurations(instances):
    for category, name, path in instances:
        yield "baseline", category, name, path, "baseline", "all", dict(BASELINE)
    base = next(item for item in instances if item[1] == "base")
    for value in POPULATION_VALUES:
        yield "population-sweep", *base, "population_size", value, {
            **BASELINE,
            "population_size": value,
        }
    for value in GENERATION_VALUES:
        yield "generation-sweep", *base, "generations", value, {
            **BASELINE,
            "generations": value,
        }
    for value in MUTATION_RATES:
        yield "mutation-sweep", *base, "mutation_rate", value, {
            **BASELINE,
            "mutation_rate": value,
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
    ax.set_xticks(list(x), [row["instance"] for row in rows], rotation=35, ha="right")
    ax.set(title="GA baseline by fixture (mean ± 1 SD, five seeds)",
           ylabel="Best-found total package value")
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
    x = [row["factor_value"] for row in rows]
    axes[0].errorbar(x, [row["final_value_mean"] for row in rows],
                     yerr=[row["final_value_stdev"] for row in rows], marker="o", capsize=4)
    axes[0].set(title="Best-found value (mean ± 1 SD)", xlabel=rows[0]["factor"], ylabel="Package value")
    axes[1].plot(x, [row["duration_seconds_mean"] for row in rows], marker="o")
    axes[1].set(title="Mean runtime", xlabel=rows[0]["factor"], ylabel="Seconds")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"GA {experiment.replace('-', ' ')} on base (five seeds)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_ga_analysis(
    fixtures_dir: str | Path = "tests/fixtures",
    output_dir: str | Path = "output/ga-analysis",
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
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            result: SearchResult[State] = genetic_algorithm(
                initial, seed=seed, run_index=run_index, **parameters
            )
            serialized = _serialize_result(result)
            violations = validate_state(result.final_state)
            run_dir = _run_directory(
                output_dir, experiment, instance, factor, factor_value, seed
            )
            record = {
                "experiment": experiment,
                "category": category,
                "instance": instance,
                "factor": factor,
                "factor_value": factor_value,
                "parameters": parameters,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "loaded_packages": sum(p is not None for p in result.final_state.placements.values()),
                "valid": not violations,
                "violations": violations,
                "evaluations": (parameters["generations"] + 1) * parameters["population_size"],
                "duration_seconds": result.duration_seconds,
                "artifact_path": str(run_dir / "result.json"),
                "result": serialized,
            }
            records.append(record)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            if experiment == "baseline" and instance not in representative_plotted:
                plot_ga_history(result, figure_dir / instance / "objective-history.png")
                plot_state_comparison(
                    result.initial_state, result.final_state,
                    figure_dir / instance / "initial-final-states.png",
                )
                representative_plotted.add(instance)
            print(f"{experiment:17s} {instance:20s} seed={seed} value={result.objective_value}", flush=True)

    summaries = summarize_ga_analysis(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"seeds": list(seeds), "baseline": BASELINE,
                    "sweeps": {"population_size": list(POPULATION_VALUES),
                               "generations": list(GENERATION_VALUES),
                               "mutation_rate": list(MUTATION_RATES)},
                    "summary": summaries}, indent=2), encoding="utf-8"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_fixture_summary(summaries, figure_dir / "fixture-summary.png")
    for experiment in ("population-sweep", "generation-sweep", "mutation-sweep"):
        _plot_parameter_sweep(summaries, experiment, figure_dir / f"{experiment}.png")
    return tuple(records)


if __name__ == "__main__":
    runs = run_ga_analysis()
    print(f"Completed {len(runs)} GA runs under output/ga-analysis")
