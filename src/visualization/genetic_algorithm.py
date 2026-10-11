"""Matplotlib visualizations for Genetic Algorithm results."""

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.model import State
from src.results import GeneticAlgorithmMetrics, SearchResult


def plot_ga_history(result: SearchResult[State], destination: str | Path) -> None:
    """Save best and average population objective against generation."""
    if not isinstance(result.metrics, GeneticAlgorithmMetrics):
        raise TypeError("result must contain GeneticAlgorithmMetrics")
    generations = range(len(result.metrics.best_objective_history))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, result.metrics.best_objective_history, label="Maximum population objective")
    ax.plot(generations, result.metrics.average_objective_history, label="Average population objective")
    ax.set(title=f"Genetic Algorithm — {result.objective_name}", xlabel="Generation", ylabel="Objective value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _box_faces(x: int, y: int, z: int, width: int, length: int, height: int):
    vertices = (
        (x, y, z),
        (x + width, y, z),
        (x + width, y + length, z),
        (x, y + length, z),
        (x, y, z + height),
        (x + width, y, z + height),
        (x + width, y + length, z + height),
        (x, y + length, z + height),
    )
    return (
        (vertices[0], vertices[1], vertices[2], vertices[3]),
        (vertices[4], vertices[5], vertices[6], vertices[7]),
        (vertices[0], vertices[1], vertices[5], vertices[4]),
        (vertices[1], vertices[2], vertices[6], vertices[5]),
        (vertices[2], vertices[3], vertices[7], vertices[6]),
        (vertices[3], vertices[0], vertices[4], vertices[7]),
    )


def _draw_state(ax, state: State, title: str) -> None:
    width, length, height = state.truck.dimensions
    corners = (
        (0, 0, 0), (width, 0, 0), (width, length, 0), (0, length, 0),
        (0, 0, height), (width, 0, height), (width, length, height), (0, length, height),
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        ax.plot(
            (corners[start][0], corners[end][0]),
            (corners[start][1], corners[end][1]),
            (corners[start][2], corners[end][2]),
            color="#45515c",
            linewidth=0.8,
        )

    outside = []
    colors = plt.get_cmap("tab20")
    for index, package in enumerate(state.packages):
        placement = state.placements[package.id]
        if placement is None:
            outside.append(package.id)
            continue
        x, y, z = placement.coordinate
        w, d, h = placement.oriented_dimensions
        ax.add_collection3d(
            Poly3DCollection(
                _box_faces(x, y, z, w, d, h),
                facecolors=colors(index % 20),
                edgecolors="#1f2933",
                linewidths=0.7,
                alpha=0.72,
            )
        )
        ax.text(x + w / 2, y + d / 2, z + h, package.id, ha="center", va="bottom", fontsize=7)

    ax.set(xlim=(0, width), ylim=(0, length), zlim=(0, height), xlabel="Width (x)", ylabel="Length (y)", zlabel="Height (z)")
    ax.set_box_aspect(state.truck.dimensions)
    ax.view_init(elev=24, azim=-55)
    outside_text = ", ".join(outside) if outside else "none"
    ax.set_title(
        f"{title}\nTotal package value: {state.objective()} | Outside: {outside_text}"
    )


def plot_state_comparison(
    initial_state: State, final_state: State, destination: str | Path
) -> None:
    """Save side-by-side 3D views of the initial and best final states."""
    if initial_state.truck != final_state.truck or initial_state.packages != final_state.packages:
        raise ValueError("states must describe the same packages and truck")
    fig = plt.figure(figsize=(13, 6))
    _draw_state(fig.add_subplot(1, 2, 1, projection="3d"), initial_state, "Initial state")
    _draw_state(fig.add_subplot(1, 2, 2, projection="3d"), final_state, "Best final state")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_ga_sweep_summary(
    summaries: Sequence[dict[str, Any]], experiment: str, destination: str | Path
) -> None:
    """Plot mean best/average histories across runs for each configuration."""
    selected = [summary for summary in summaries if summary["experiment"] == experiment]
    if not selected:
        raise ValueError(f"no GA summaries found for experiment {experiment!r}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10")
    for index, summary in enumerate(selected):
        if experiment == "generation-sweep":
            label = f"generations={summary['generations']}"
        elif experiment == "population-sweep":
            label = f"population={summary['population_size']}"
        else:
            raise ValueError(f"unknown GA sweep {experiment!r}")
        color = colors(index % 10)
        x = range(len(summary["best_history_mean"]))
        ax.plot(x, summary["best_history_mean"], color=color, label=f"{label} — max")
        ax.plot(
            x,
            summary["average_history_mean"],
            color=color,
            linestyle="--",
            label=f"{label} — average",
        )
        best = summary["best_history_mean"]
        spread = summary["best_history_stdev"]
        ax.fill_between(
            x,
            [value - deviation for value, deviation in zip(best, spread)],
            [value + deviation for value, deviation in zip(best, spread)],
            color=color,
            alpha=0.12,
        )

    ax.set(
        title=f"GA {experiment.replace('-', ' ')} — mean across runs",
        xlabel="Generation",
        ylabel="Objective value",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
