"""One-factor-at-a-time ablation of the three GA operators."""

import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.experiments.genetic_algorithm import _serialize_result
from src.model import Package, Truck
from src.neighbors import random_initial_state
from src.visualization.genetic_algorithm import plot_ablation_comparison


def run_ga_ablation(
    packages: tuple[Package, ...],
    truck: Truck,
    *,
    seeds: tuple[int, ...] = (41, 42, 43, 44, 45),
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.05,
    output_dir: str | Path = "output/ga-ablation",
) -> tuple[dict[str, Any], ...]:
    if len(seeds) < 2 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two distinct integers")
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    if type(generations) is not int or generations < 0:
        raise ValueError("generations must be a non-negative integer")
    if isinstance(mutation_rate, bool) or not isinstance(mutation_rate, (int, float)) or not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    ablations = (
        ("baseline", mutation_rate, "roulette", True),
        ("no-mutation", 0.0, "roulette", True),
        ("no-crossover", mutation_rate, "roulette", False),
        ("uniform-selection", mutation_rate, "uniform", True),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "runs.json").exists() or (output_dir / "runs.jsonl").exists():
        raise FileExistsError(f"results already exist in {output_dir}; choose a new output directory")
    records = []
    total = len(ablations) * len(seeds)
    for variant_index, (variant, rate, selection, crossover) in enumerate(ablations):
        for seed_index, seed in enumerate(seeds, start=1):
            index = variant_index * len(seeds) + seed_index
            print(
                f"Run {index}/{total}: {variant}, seed={seed}, population={population_size}, "
                f"generations={generations}, mutation_rate={rate}",
                flush=True,
            )
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            result = genetic_algorithm(
                initial,
                seed=seed,
                population_size=population_size,
                generations=generations,
                mutation_rate=rate,
                selection_mode=selection,
                crossover_enabled=crossover,
                run_index=seed_index,
            )
            record = {
                "variant": variant,
                "seed": seed,
                "initial_state_seed": initial_seed,
                "parameters": {
                    "population_size": population_size,
                    "generations": generations,
                    "mutation_rate": rate,
                    "selection_mode": selection,
                    "crossover_enabled": crossover,
                },
                "result": _serialize_result(result),
            }
            records.append(record)
            with (output_dir / "runs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    summaries = []
    for variant, *_ in ablations:
        runs = [record["result"] for record in records if record["variant"] == variant]
        best = [run["metrics"]["best_objective_history"] for run in runs]
        average = [run["metrics"]["average_objective_history"] for run in runs]
        objectives = [run["objective_value"] for run in runs]
        summaries.append({
            "variant": variant,
            "run_count": len(runs),
            "final_objective_mean": mean(objectives),
            "final_objective_stdev": stdev(objectives),
            "duration_seconds_mean": mean(run["duration_seconds"] for run in runs),
            "best_history_mean": [mean(values[i] for values in best) for i in range(generations + 1)],
            "average_history_mean": [mean(values[i] for values in average) for i in range(generations + 1)],
        })
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_ablation_comparison(summaries, output_dir / "objective-comparison.png")
    return tuple(records)
