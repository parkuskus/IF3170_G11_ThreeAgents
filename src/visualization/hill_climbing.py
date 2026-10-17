from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.model import State
from src.results import HillClimbingMetrics, SearchResult


def _save(fig, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_hc_history(result: SearchResult[State], destination: str | Path) -> None:
    metrics = result.metrics
    if not isinstance(metrics, HillClimbingMetrics):
        raise TypeError("result harus berisi HillClimbingMetrics")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(result.objective_history)), result.objective_history)
    if metrics.iterations_per_restart:
        for boundary in list(accumulate(n + 1 for n in metrics.iterations_per_restart))[:-1]:
            ax.axvline(boundary, color="#c0392b", linestyle=":", linewidth=0.8)
    ax.set(
        title=f"Hill-climbing ({metrics.variant}) — {result.objective_name}",
        xlabel="Iteration (restart boundaries dotted)" if metrics.iterations_per_restart else "Iteration",
        ylabel="Objective value",
    )
    ax.grid(True, alpha=0.25)
    _save(fig, destination)


def plot_hc_summary(summaries: Sequence[dict[str, Any]], destination: str | Path) -> None:
    if not summaries:
        raise ValueError("minimal satu ringkasan diperlukan")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [s["variant"] for s in summaries],
        [s["objective_mean"] for s in summaries],
        yerr=[s["objective_stdev"] for s in summaries],
        capsize=4,
        color="#4c72b0",
    )
    ax.set(title="Hill-climbing variants — final objective (mean ± SD)", ylabel="Objective value")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)
