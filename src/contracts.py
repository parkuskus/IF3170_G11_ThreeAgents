"""Shared invocation contract for Hill-climbing, SA, and GA implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from src.results import ObjectiveValue, ParameterValue, SearchResult


StateT = TypeVar("StateT")
ObjectiveFunction: TypeAlias = Callable[[StateT], ObjectiveValue]


@dataclass(frozen=True, slots=True)
class SearchProblem(Generic[StateT]):
    """Domain operations shared by local-search algorithms."""

    objective_fn: ObjectiveFunction[StateT]
    is_valid: Callable[[StateT], bool]
    random_state: Callable[[Random], StateT]
    neighbor: Callable[[StateT, Random], StateT]
    crossover: Callable[[StateT, StateT, Random], StateT] | None = None
    mutate: Callable[[StateT, Random], StateT] | None = None
    objective_name: str = "custom"

    def __post_init__(self) -> None:
        for name in ("objective_fn", "is_valid", "random_state", "neighbor"):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")
        for name in ("crossover", "mutate"):
            operation = getattr(self, name)
            if operation is not None and not callable(operation):
                raise ValueError(f"{name} must be callable or None")
        if not isinstance(self.objective_name, str) or not self.objective_name:
            raise ValueError("objective_name must be a non-empty string")


class _ObjectiveState(Protocol):
    def objective(self) -> ObjectiveValue: ...


class SearchAlgorithm(Protocol[StateT]):
    """Callable interface shared by every search algorithm.

    Implementations accept algorithm-specific keyword parameters and return a
    SearchResult with the supplied seed and effective parameters recorded.
    """

    def __call__(
        self,
        initial_state: StateT,
        *,
        seed: int,
        objective_fn: ObjectiveFunction[StateT] | None = None,
        search_problem: SearchProblem[StateT] | None = None,
        **parameters: ParameterValue,
    ) -> SearchResult[StateT]: ...


def resolve_objective(
    state: StateT, objective_fn: ObjectiveFunction[StateT] | None = None
) -> ObjectiveFunction[StateT]:
    """Use the supplied objective, or the state's built-in objective method."""
    if objective_fn is not None:
        return objective_fn
    default = getattr(state, "objective", None)
    if not callable(default):
        raise TypeError("objective_fn is required when state has no objective() method")
    return lambda candidate: cast(_ObjectiveState, candidate).objective()
