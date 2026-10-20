from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_benchmark_summary(summaries: Sequence[dict[str, Any]], destination: str | Path) -> None:
    """Grouped bars: mean final value as a share of the best known value, per instance and algorithm."""
    instances = list(dict.fromkeys(s["instance"] for s in summaries if s["best_known"] > 0))
    algorithms = list(dict.fromkeys(s["algorithm"] for s in summaries))
    lookup = {(s["instance"], s["algorithm"]): s for s in summaries}
    width = 0.8 / len(algorithms)
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, algorithm in enumerate(algorithms):
        shares = [
            lookup[(instance, algorithm)]["final_value_mean"] / lookup[(instance, algorithm)]["best_known"]
            for instance in instances
        ]
        positions = [i + (index - (len(algorithms) - 1) / 2) * width for i in range(len(instances))]
        ax.bar(positions, shares, width, label=algorithm)
    ax.set_xticks(range(len(instances)), instances, rotation=30, ha="right")
    ax.set(ylabel="Mean final value / best known", ylim=(0, 1.08), title="Benchmark per instance and algorithm")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_fleet_benchmark_summary(
    summaries: Sequence[dict[str, Any]], destination: str | Path
) -> None:
    """Compare methods per fixture, showing spread only when repeated runs exist."""
    if not summaries:
        raise ValueError("at least one fleet benchmark summary is required")

    fixtures = list(dict.fromkeys(row["instance"] for row in summaries))
    methods = list(dict.fromkeys(row["method"] for row in summaries))
    lookup = {(row["instance"], row["method"]): row for row in summaries}
    show_spread = any(row.get("runs", 1) > 1 for row in summaries)
    columns = min(4, len(fixtures))
    rows = (len(fixtures) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.25 * columns, 3.7 * rows), squeeze=False)
    colors = plt.get_cmap("tab10")

    for axis, fixture in zip(axes.flat, fixtures):
        rows = [lookup[(fixture, method)] for method in methods]
        bars = {
            "color": [colors(index % 10) for index in range(len(rows))],
        }
        if show_spread:
            bars.update(yerr=[row["final_value_stdev"] for row in rows], capsize=3)
        axis.bar(range(len(rows)), [row["final_value_mean"] for row in rows], **bars)
        axis.set_title(fixture)
        axis.set_xticks(range(len(rows)), methods, rotation=55, ha="right", fontsize=7)
        axis.grid(True, axis="y", alpha=0.25)

    for axis in axes.flat[len(fixtures):]:
        axis.set_visible(False)
    spread_label = "mean ± 1 SD" if show_spread else "single observation"
    fig.supylabel(f"Final total package value ({spread_label})")
    fig.suptitle("Multi-truck benchmark by fixture")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_fleet_transfer(record: dict[str, Any], destination: str | Path) -> None:
    """Show two reciprocal cross-truck assignments in recorded initial/final states."""
    before, after = record["initial_state"], record["final_state"]
    if before["truck_count"] != after["truck_count"]:
        raise ValueError("fleet size changed between states")
    assignments = [
        {package["id"]: index for index, truck in enumerate(state["trucks"])
         for package in truck["packages"]}
        for state in (before, after)
    ]
    moved = [
        package_id for package_id in assignments[0].keys() & assignments[1].keys()
        if assignments[0][package_id] != assignments[1][package_id]
    ]
    if (assignments[0].keys() != assignments[1].keys() or len(moved) != 2
            or assignments[0][moved[0]] != assignments[1][moved[1]]
            or assignments[0][moved[1]] != assignments[1][moved[0]]):
        raise ValueError("states must contain a reciprocal cross-truck exchange")

    colors = dict(zip(sorted(moved), ("#bf5d26", "#087f8c")))
    count = before["truck_count"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.suptitle(
        f"tight-capacity | HC sideways | {count} truk | seed {record['seed']}\n"
        f"Nilai {record['initial_value']} menjadi {record['final_value']}",
        fontsize=14,
    )
    for state, bottom, heading in ((before, 0.66, "SEBELUM"), (after, 0.11, "SESUDAH")):
        ax.text(0.01, bottom + 0.27, heading, fontsize=11, weight="bold", transform=ax.transAxes)
        for index, truck in enumerate(state["trucks"]):
            center = (index + 0.5) / count
            width = 0.78 / count
            ax.add_patch(FancyBboxPatch(
                (center - width / 2, bottom), width, 0.25,
                boxstyle="round,pad=0.01", facecolor="#f4f6f8", edgecolor="#aeb9c3",
                transform=ax.transAxes,
            ))
            ax.text(center, bottom + 0.205, f"TRUK {index + 1}", ha="center", weight="bold",
                    fontsize=10, transform=ax.transAxes)
            packages = truck["packages"]
            for package_index, package in enumerate(packages):
                package_id = package["id"]
                ax.text(
                    center, bottom + 0.17 - package_index * 0.045, package_id,
                    ha="center", va="center", fontsize=9, weight="bold",
                    color="white" if package_id in colors else "#2a3945",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": colors.get(package_id, "#dce3e8"),
                          "edgecolor": "none"},
                    transform=ax.transAxes,
                )

    for offset, package_id in enumerate(sorted(moved)):
        origin = (assignments[0][package_id] + 0.5) / count
        target = (assignments[1][package_id] + 0.5) / count
        ax.annotate(
            "", xy=(target, 0.39), xytext=(origin, 0.64), xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": colors[package_id],
                        "connectionstyle": f"arc3,rad={-0.14 if offset == 0 else 0.14}"},
        )
    for offset, package_id in enumerate(sorted(moved)):
        ax.text(0.5, 0.57 - offset * 0.13, f"{package_id}  Truk {assignments[0][package_id] + 1}"
                f" ke Truk {assignments[1][package_id] + 1}", ha="center", fontsize=9,
                color=colors[package_id], weight="bold", transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3}, zorder=5)
    ax.text(0.5, 0.025, "State awal dan akhir, bukan rekaman satu langkah pencarian",
            ha="center", fontsize=9, color="#52616c", transform=ax.transAxes)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
