
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.experiments.common.benchmark import discover_instances
from src.experiments.common.data import load_problem
from src.experiments.common.objective_comparison import resolve_objective_fn
from src.experiments.hc.experiments import _serialize_result
from src.model import State
from src.neighbors import random_initial_state
from src.results import SearchResult
from src.validation import validate_state
from src.visualization.genetic_algorithm import plot_state_comparison
from src.visualization.hill_climbing import plot_hc_history


SEEDS = (11, 12, 13, 14, 15)
BASELINE = {
    "neighbors_per_iteration": 30,
    "max_iterations": 1000,
    "max_sideways": 100,
    "max_restarts": 10,
}

SWEEPS = {
    "neighbors-sweep": (
        "neighbors_per_iteration", (10, 30, 60),
        ("steepest", "sideways", "random_restart"),
    ),
    "iterations-sweep": (
        "max_iterations", (500, 1000, 2000),
        tuple(VARIANTS),
    ),
    "sideways-sweep": ("max_sideways", (50, 100, 300), ("sideways",)),
    "restarts-sweep": ("max_restarts", (5, 10, 20), ("random_restart",)),
}


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_hc_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["experiment"], record["instance"], record["variant"],
               record["factor"], str(record["factor_value"]))
        groups[key].append(record)

    summaries = []
    for (experiment, instance, variant, factor, value), runs in sorted(groups.items()):
        objectives = [run["result"]["objective_value"] for run in runs]
        summaries.append(
            {
                "experiment": experiment,
                "instance": instance,
                "variant": variant,
                "factor": factor,
                "factor_value": "all" if factor == "baseline" else int(value),
                "run_count": len(runs),
                "seeds": [run["seed"] for run in runs],
                "valid_rate": mean(1.0 if run["valid"] else 0.0 for run in runs),
                "initial_value_mean": mean(run["initial_value"] for run in runs),
                "final_value_mean": mean(objectives),
                "final_value_stdev": _spread(objectives),
                "final_value_best": max(objectives),
                "loaded_packages_mean": mean(run["loaded_packages"] for run in runs),
                "iterations_mean": mean(run["iterations"] for run in runs),
                "sideways_moves_mean": mean(run["sideways_moves"] for run in runs),
                "restarts_mean": mean(run["restarts"] for run in runs),
                "duration_seconds_mean": mean(run["duration_seconds"] for run in runs),
            }
        )
    return summaries


def _configurations(instances):
    for category, name, path in instances:
        for variant in VARIANTS:
            yield "baseline", category, name, path, variant, "baseline", "all", dict(BASELINE)
    base = next(item for item in instances if item[1] == "base")
    for experiment, (factor, values, variants) in SWEEPS.items():
        for variant in variants:
            for value in values:
                yield experiment, *base, variant, factor, value, {**BASELINE, factor: value}


def _run_directory(output_dir: Path, experiment: str, instance: str, variant: str,
                   factor: str, factor_value: int | str, seed: int) -> Path:
    return (output_dir / experiment / instance / variant
            / f"{factor}-{factor_value}" / f"seed-{seed}")


def _plot_fixture_summary(summaries, destination: Path) -> None:
    rows = [r for r in summaries if r["experiment"] == "baseline"]
    instances = list(dict.fromkeys(r["instance"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, variant in enumerate(variants):
        by_instance = {r["instance"]: r for r in rows if r["variant"] == variant}
        xs = [j + i * width for j in range(len(instances))]
        ax.bar(xs, [by_instance[n]["final_value_mean"] for n in instances], width,
               yerr=[by_instance[n]["final_value_stdev"] for n in instances],
               capsize=2, label=variant)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(instances))], instances,
                  rotation=35, ha="right")
    ax.set(title="HC baseline by fixture (mean ± 1 SD, five seeds)",
           ylabel="Best-found total package value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _plot_parameter_sweep(summaries, experiment: str, destination: Path) -> None:
    rows = [r for r in summaries if r["experiment"] == experiment]
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for variant in variants:
        vr = sorted((r for r in rows if r["variant"] == variant),
                    key=lambda r: r["factor_value"])
        x = [str(r["factor_value"]) for r in vr]
        axes[0].errorbar(x, [r["final_value_mean"] for r in vr],
                         yerr=[r["final_value_stdev"] for r in vr],
                         marker="o", capsize=4, label=variant)
        axes[1].plot(x, [r["duration_seconds_mean"] for r in vr], marker="o", label=variant)
    factor = rows[0]["factor"]
    axes[0].set(title="Best-found value (mean ± 1 SD)", xlabel=factor, ylabel="Package value")
    axes[1].set(title="Duration (mean)", xlabel=factor, ylabel="Seconds")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"HC {experiment.replace('-', ' ')} on base (five seeds)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_hc_analysis(
    fixtures_dir: str | Path = "tests/fixtures",
    output_dir: str | Path = "output/hc-analysis",
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[dict[str, Any], ...]:
    """Run five-seed fixture baselines and paired one-factor sweeps on base."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("seeds must contain at least two distinct integers")
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    objective_fn = resolve_objective_fn(None, "total_value")
    records = []
    representative_plotted = set()
    instances = discover_instances(Path(fixtures_dir))
    for (experiment, category, instance, path, variant,
         factor, factor_value, parameters) in _configurations(instances):
        packages, truck = load_problem(path)
        for run_index, seed in enumerate(seeds, 1):
            # Same derivation as the SA/GA runners and benchmark: identical initial states.
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            result: SearchResult[State] = hill_climbing(
                initial, seed=seed, variant=variant, run_index=run_index,
                objective_fn=objective_fn, objective_name="total_value", **parameters,
            )
            violations = validate_state(result.final_state)
            run_dir = _run_directory(output_dir, experiment, instance, variant,
                                     factor, factor_value, seed)
            metrics = result.metrics
            record = {
                "experiment": experiment,
                "category": category,
                "instance": instance,
                "variant": variant,
                "factor": factor,
                "factor_value": factor_value,
                "parameters": parameters,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "initial_value": result.initial_state.objective(),
                "loaded_packages": sum(p is not None for p in result.final_state.placements.values()),
                "valid": not violations,
                "violations": violations,
                "iterations": metrics.iterations,
                "sideways_moves": metrics.sideways_moves,
                "restarts": metrics.restarts,
                "duration_seconds": result.duration_seconds,
                "artifact_path": str(run_dir / "result.json"),
                "result": _serialize_result(result),
            }
            records.append(record)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            plot_key = (instance, variant)
            if experiment == "baseline" and plot_key not in representative_plotted:
                plot_hc_history(result, figure_dir / instance / variant / "objective-history.png")
                plot_state_comparison(
                    result.initial_state, result.final_state,
                    figure_dir / instance / variant / "initial-final-states.png",
                )
                representative_plotted.add(plot_key)
            print(f"{experiment:17s} {instance:18s} {variant:14s} seed={seed} "
                  f"value={result.objective_value}", flush=True)

    summaries = summarize_hc_analysis(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"seeds": list(seeds), "baseline": BASELINE,
                    "sweeps": {exp: {"factor": f, "values": list(v), "variants": list(vs)}
                               for exp, (f, v, vs) in SWEEPS.items()},
                    "summary": summaries}, indent=2), encoding="utf-8"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_fixture_summary(summaries, figure_dir / "fixture-summary.png")
    for experiment in SWEEPS:
        _plot_parameter_sweep(summaries, experiment, figure_dir / f"{experiment}.png")
    return tuple(records)


if __name__ == "__main__":
    runs = run_hc_analysis()
    print(f"Completed {len(runs)} HC runs under output/hc-analysis")