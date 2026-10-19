from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
