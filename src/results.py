"""Shared return types for local-search algorithm runs."""

from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeAlias, TypeVar


StateT = TypeVar("StateT")
ObjectiveValue: TypeAlias = int | float
ParameterValue: TypeAlias = int | float | str | bool


@dataclass(frozen=True, slots=True)
class HillClimbingMetrics:
    """Metrics shared by steepest, stochastic, sideways, and restart variants."""

    variant: str
    iterations: int
    sideways_moves: int = 0
    maximum_sideways_moves: int | None = None
    restarts: int = 0
    maximum_restarts: int | None = None
    iterations_per_restart: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SimulatedAnnealingMetrics:
    iterations: int
    exp_delta_e_over_temperature: tuple[float, ...] = ()
    stuck_count: int = 0


@dataclass(frozen=True, slots=True)
class GeneticAlgorithmMetrics:
    generations: int
    population_size: int
    best_objective_history: tuple[ObjectiveValue, ...] = ()
    average_objective_history: tuple[float, ...] = ()


AlgorithmMetrics: TypeAlias = (
    HillClimbingMetrics | SimulatedAnnealingMetrics | GeneticAlgorithmMetrics
)


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[StateT]):
    """One run's common output plus algorithm-specific metrics.

    GA should seed its initial population with initial_state; final_state is
    the best solution found. StateT also permits a future multi-truck state.
    """

    algorithm: str
    objective_name: str
    initial_state: StateT
    final_state: StateT
    objective_value: ObjectiveValue
    objective_history: tuple[ObjectiveValue, ...]
    duration_seconds: float
    metrics: AlgorithmMetrics
    seed: int | None = None
    run_index: int = 1
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("algorithm name must not be empty")
        if not self.objective_name:
            raise ValueError("objective name must not be empty")
        if self.duration_seconds < 0:
            raise ValueError("duration must be non-negative")
        if self.run_index < 1:
            raise ValueError("run index must be positive")
