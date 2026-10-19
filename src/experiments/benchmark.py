"""Cross-algorithm benchmark over the fixture instances (paired seeds and initial states)."""

import json
from collections.abc import Sequence
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.data import load_problem
from src.model import State
from src.neighbors import random_initial_state
from src.validation import validate_state
from src.visualization.benchmark import plot_benchmark_summary

# (category, instance name, path relative to the fixtures directory)
CATEGORIES = {
    "base": "Kasus dasar",
    "tight": "Kasus kendala ketat",
    "infeasible": "Kasus tanpa pemuatan nonkosong yang layak",
    "boundary": "Kasus batas dan struktur khusus",
}

HC_PARAMETERS = dict(neighbors_per_iteration=30, max_iterations=1000, max_sideways=100, max_restarts=10)
SA_PARAMETERS = dict(max_iterations=3000, initial_temperature=100.0, cooling_rate=0.995, stuck_patience=100)
GA_PARAMETERS = dict(population_size=20, generations=40, mutation_rate=0.05)


def discover_instances(fixtures_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (category, name, path) for every valid-instance fixture."""
    instances = [("base", "base", fixtures_dir / "base.json")]
    instances += [("tight", name, fixtures_dir / f"{name}.json") for name in ("tight-space", "tight-capacity")]
    instances.append(("infeasible", "infeasible", fixtures_dir / "infeasible.json"))
    instances += [
        ("boundary", path.stem, path) for path in sorted((fixtures_dir / "boundary").glob("*.json"))
    ]
    return instances


def check_invalid_inputs(fixtures_dir: Path) -> list[dict[str, Any]]:
    """Feed every invalid fixture to the loader; each must be rejected with ValueError."""
    rows = []
    for path in sorted((fixtures_dir / "invalid").glob("*.json")):
        try:
            load_problem(path)
            rows.append({"fixture": path.stem, "rejected": False, "error": None})
        except ValueError as error:
            rows.append({"fixture": path.stem, "rejected": True, "error": str(error)})
    return rows


class _CountingObjective:
    def __init__(self) -> None:
        self.evaluations = 0

    def __call__(self, state: State) -> int:
        self.evaluations += 1
        return state.objective()


def _algorithms() -> dict[str, Any]:
    algorithms: dict[str, Any] = {}
    for variant in VARIANTS:
        algorithms[f"hc_{variant}"] = lambda initial, seed, fn, v=variant: hill_climbing(
            initial, seed=seed, variant=v, objective_fn=fn, **HC_PARAMETERS
        )
    algorithms["sa"] = lambda initial, seed, fn: simulated_annealing(
        initial, seed=seed, objective_fn=fn, **SA_PARAMETERS
    )
    algorithms["ga"] = lambda initial, seed, fn: genetic_algorithm(
        initial, seed=seed, objective_fn=fn, **GA_PARAMETERS
    )
    return algorithms


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_benchmark(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["instance"], record["algorithm"]), []).append(record)
    best_known: dict[str, float] = {}
    for record in records:
        best_known[record["instance"]] = max(best_known.get(record["instance"], 0), record["final_value"])

    summaries = []
    for (instance, algorithm), rows in groups.items():
        values = [row["final_value"] for row in rows]
        summaries.append(
            {
                "category": rows[0]["category"],
                "instance": instance,
                "algorithm": algorithm,
                "runs": len(rows),
                "valid_rate": mean(1.0 if row["valid"] else 0.0 for row in rows),
                "final_value_mean": mean(values),
                "final_value_stdev": _spread(values),
                "final_value_best": max(values),
                "best_known": best_known[instance],
                "hit_best_known_rate": mean(1.0 if value == best_known[instance] else 0.0 for value in values),
                "loaded_packages_mean": mean(row["loaded_packages"] for row in rows),
                "initial_value_mean": mean(row["initial_value"] for row in rows),
                "evaluations_mean": mean(row["evaluations"] for row in rows),
                "duration_seconds_mean": mean(row["duration_seconds"] for row in rows),
            }
        )
    return summaries


def run_benchmark(
    fixtures_dir: str | Path = "tests/fixtures",
    *,
    seeds: Sequence[int] = (11, 12, 13, 14, 15),
    output_dir: str | Path = "output/benchmark",
) -> tuple[dict[str, Any], ...]:
    """Run HC (4 variants), SA, and GA on every fixture with shared seeds and initial states."""
    if len(seeds) < 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least three distinct integers")
    fixtures_dir, output_dir = Path(fixtures_dir), Path(output_dir)
    algorithms = _algorithms()
    records: list[dict[str, Any]] = []
    for category, instance, path in discover_instances(fixtures_dir):
        packages, truck = load_problem(path)
        for seed in seeds:
            # Same derivation as the per-algorithm runners: paired initial state for all algorithms.
            initial_seed = seed ^ 0x5EED
            initial = random_initial_state(packages, truck, Random(initial_seed))
            for name, run in algorithms.items():
                counter = _CountingObjective()
                result = run(initial, seed, counter)
                errors = validate_state(result.final_state)
                loaded = sum(1 for p in result.final_state.placements.values() if p is not None)
                records.append(
                    {
                        "category": category,
                        "instance": instance,
                        "algorithm": name,
                        "seed": seed,
                        "initial_state_seed": initial_seed,
                        "initial_value": initial.objective(),
                        "final_value": result.final_state.objective(),
                        "loaded_packages": loaded,
                        "package_count": len(packages),
                        "valid": not errors,
                        "violations": errors,
                        "evaluations": counter.evaluations,
                        "duration_seconds": result.duration_seconds,
                        "objective_history": list(result.objective_history),
                    }
                )
                print(f"{instance:20s} seed={seed} {name:20s} value={records[-1]['final_value']}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_benchmark(records)
    invalid = check_invalid_inputs(fixtures_dir)
    parameters = {"hc": HC_PARAMETERS, "sa": SA_PARAMETERS, "ga": GA_PARAMETERS, "seeds": list(seeds)}
    (output_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"parameters": parameters, "summary": summaries, "invalid_inputs": invalid}, indent=2),
        encoding="utf-8",
    )
    plot_benchmark_summary(summaries, output_dir / "summary.png")
    return tuple(records)
