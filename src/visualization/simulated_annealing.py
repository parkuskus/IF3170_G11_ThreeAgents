"""Matplotlib visualizations for Simulated Annealing results."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.model import State
from src.results import SearchResult, SimulatedAnnealingMetrics


def _sa_metrics(result: SearchResult[State]) -> SimulatedAnnealingMetrics:
    if not isinstance(result.metrics, SimulatedAnnealingMetrics):
        raise TypeError("result must contain SimulatedAnnealingMetrics")
    return result.metrics


def _save(fig, destination: str | Path) -> None:
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_sa_history(result: SearchResult[State], destination: str | Path) -> None:
    """Save current and best-so-far objective against iteration."""
    metrics = _sa_metrics(result)
    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = range(len(result.objective_history))
    if metrics.current_objective_history:
        ax.plot(
            iterations,
            metrics.current_objective_history,
            color="#9aa5b1",
            linewidth=0.8,
            label="Current state objective",
        )
    ax.plot(iterations, result.objective_history, color="#1f6feb", label="Best-so-far objective")
    ax.set(
        title=(
            f"Simulated Annealing — {result.objective_name} (run {result.run_index})\n"
            f"Stuck in local optima: {metrics.stuck_count}× "
            f"(patience {metrics.stuck_patience})"
        ),
        xlabel="Iteration",
        ylabel="Objective value",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, destination)


def plot_sa_acceptance(result: SearchResult[State], destination: str | Path) -> None:
    """Save e^(ΔE/T) per iteration, with the temperature schedule on a twin axis."""
    metrics = _sa_metrics(result)
    iterations = range(1, len(metrics.exp_delta_e_over_temperature) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        iterations,
        metrics.exp_delta_e_over_temperature,
        s=4,
        color="#d1495b",
        alpha=0.55,
        label="e^(ΔE/T), capped at 1",
    )
    ax.set(
        title=f"Simulated Annealing acceptance probability (run {result.run_index})",
        xlabel="Iteration",
        ylabel="e^(ΔE/T)",
        ylim=(-0.03, 1.03),
    )
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if metrics.temperature_history:
        temperature_ax = ax.twinx()
        temperature_ax.plot(
            iterations, metrics.temperature_history, color="#2e4057", linewidth=1, label="Temperature T"
        )
        temperature_ax.set_ylabel("Temperature T")
        extra_handles, extra_labels = temperature_ax.get_legend_handles_labels()
        handles += extra_handles
        labels += extra_labels
    ax.legend(handles, labels, loc="center right")
    _save(fig, destination)


def plot_sa_runs_summary(
    results: Sequence[SearchResult[State]], destination: str | Path
) -> None:
    """Overlay the best-so-far objective histories of repeated SA runs."""
    if not results:
        raise ValueError("at least one SA result is required")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10")
    for index, result in enumerate(results):
        ax.plot(
            range(len(result.objective_history)),
            result.objective_history,
            color=colors(index % 10),
            label=f"run {result.run_index} (seed {result.seed}) — final {result.objective_value:g}",
        )
    ax.set(
        title=f"Simulated Annealing — {results[0].objective_name}, best-so-far per run",
        xlabel="Iteration",
        ylabel="Objective value",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, destination)


def plot_objective_comparison(
    comparisons: Sequence[dict[str, Any]],
    destination: str | Path,
    title: str = "SA final total package value by objective function (mean ± 1 SD)",
) -> None:
    """Bar chart of the final total package value reached under each objective."""
    if not comparisons:
        raise ValueError("at least one objective comparison is required")
    names = [comparison["objective_name"] for comparison in comparisons]
    means = [comparison["total_package_value_mean"] for comparison in comparisons]
    spreads = [comparison["total_package_value_stdev"] for comparison in comparisons]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=spreads, capsize=6, color="#1f6feb", alpha=0.8)
    ax.bar_label(bars, labels=[f"{value:g}" for value in means], padding=3)
    ax.set(
        title=title,
        xlabel="Objective function optimised",
        ylabel="Total package value in truck",
    )
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)
