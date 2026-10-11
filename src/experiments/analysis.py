"""Aggregate repeated Genetic Algorithm runs for analysis and plots."""

from collections import defaultdict
from statistics import mean, stdev
from typing import Any


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_ga_runs(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Group runs by sweep configuration and compute means/stddevs."""
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["experiment"],
            record["population_size"],
            record["generations"],
        )
        groups[key].append(record)

    summaries = []
    for (experiment, population_size, generations), runs in sorted(groups.items()):
        results = [run["result"] for run in runs]
        best_histories = [result["metrics"]["best_objective_history"] for result in results]
        average_histories = [result["metrics"]["average_objective_history"] for result in results]
        if len({len(history) for history in best_histories + average_histories}) != 1:
            raise ValueError("runs in a configuration must have matching history lengths")

        summaries.append(
            {
                "experiment": experiment,
                "population_size": population_size,
                "generations": generations,
                "run_count": len(runs),
                "seeds": [run["seed"] for run in runs],
                "final_objective_mean": mean(result["objective_value"] for result in results),
                "final_objective_stdev": _spread(
                    [result["objective_value"] for result in results]
                ),
                "duration_seconds_mean": mean(
                    result["duration_seconds"] for result in results
                ),
                "duration_seconds_stdev": _spread(
                    [result["duration_seconds"] for result in results]
                ),
                "best_history_mean": [
                    mean(history[i] for history in best_histories)
                    for i in range(len(best_histories[0]))
                ],
                "best_history_stdev": [
                    _spread([history[i] for history in best_histories])
                    for i in range(len(best_histories[0]))
                ],
                "average_history_mean": [
                    mean(history[i] for history in average_histories)
                    for i in range(len(average_histories[0]))
                ],
                "average_history_stdev": [
                    _spread([history[i] for history in average_histories])
                    for i in range(len(average_histories[0]))
                ],
            }
        )
    return tuple(summaries)
