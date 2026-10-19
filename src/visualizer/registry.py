"""Algorithm catalogue: callable plus a parameter spec the UI renders its forms from."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.objectives import OBJECTIVES


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    label: str
    kind: str  # "int" | "float" | "bool" | "select"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: tuple[str, ...] = ()
    help: str = ""
    # Only shown when another parameter has one of these values: (param, values).
    visible_when: tuple[str, tuple[str, ...]] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "options": list(self.options),
            "help": self.help,
            "visibleWhen": (
                {"param": self.visible_when[0], "values": list(self.visible_when[1])}
                if self.visible_when
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    key: str
    label: str
    run: Callable[..., Any]
    params: tuple[Param, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "params": [param.to_json() for param in self.params],
        }

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Validate raw UI values against the spec and fill in defaults."""
        known = {param.name: param for param in self.params}
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"unknown parameters for {self.key}: {sorted(unknown)}")
        values = {}
        for name, param in known.items():
            value = raw.get(name, param.default)
            if param.kind == "int":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
                    raise ValueError(f"{name} must be an integer")
                value = int(value)
            elif param.kind == "float":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be a number")
                value = float(value)
            elif param.kind == "bool":
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
            elif param.kind == "select":
                if value not in param.options:
                    raise ValueError(f"{name} must be one of {list(param.options)}")
            values[name] = value
        return values


ALGORITHMS: dict[str, AlgorithmSpec] = {
    spec.key: spec
    for spec in (
        AlgorithmSpec(
            "hill_climbing",
            "Hill-climbing",
            hill_climbing,
            (
                Param("variant", "Variant", "select", "steepest", options=tuple(VARIANTS)),
                Param(
                    "neighbors_per_iteration", "Neighbors per iteration", "int", 30,
                    minimum=1, step=1,
                    help="Ignored by the stochastic variant (always one neighbor).",
                    visible_when=("variant", ("steepest", "sideways", "random_restart")),
                ),
                Param("max_iterations", "Max iterations", "int", 1000, minimum=0, step=100),
                Param(
                    "max_sideways", "Max sideways moves", "int", 100, minimum=0, step=10,
                    visible_when=("variant", ("sideways",)),
                ),
                Param(
                    "max_restarts", "Max restarts", "int", 10, minimum=0, step=1,
                    visible_when=("variant", ("random_restart",)),
                ),
            ),
        ),
        AlgorithmSpec(
            "simulated_annealing",
            "Simulated Annealing",
            simulated_annealing,
            (
                Param("max_iterations", "Max iterations", "int", 5000, minimum=0, step=500),
                Param("initial_temperature", "Initial temperature", "float", 100.0, minimum=0.001, step=10),
                Param(
                    "cooling_rate", "Cooling rate", "float", 0.995, minimum=0.001, maximum=1, step=0.001,
                    help="Temperature is multiplied by this each iteration.",
                ),
                Param(
                    "stuck_patience", "Stuck patience", "int", 100, minimum=1, step=10,
                    help="Iterations without a new best before it counts as stuck.",
                ),
            ),
        ),
        AlgorithmSpec(
            "genetic_algorithm",
            "Genetic Algorithm",
            genetic_algorithm,
            (
                Param("population_size", "Population size", "int", 50, minimum=1, step=10),
                Param("generations", "Generations", "int", 100, minimum=0, step=10),
                Param("mutation_rate", "Mutation rate", "float", 0.05, minimum=0, maximum=1, step=0.01),
                Param("selection_mode", "Selection", "select", "roulette", options=("roulette", "uniform")),
                Param("crossover_enabled", "Crossover", "bool", True),
            ),
        ),
    )
}

OBJECTIVE_LABELS = {
    "total_value": "Total value",
    "eta_weighted_value": "ETA-weighted value",
    "compact_value": "Compact value",
}


def catalogue() -> dict[str, Any]:
    return {
        "algorithms": [spec.to_json() for spec in ALGORITHMS.values()],
        "objectives": [
            {"key": key, "label": OBJECTIVE_LABELS.get(key, key)} for key in OBJECTIVES
        ],
    }
