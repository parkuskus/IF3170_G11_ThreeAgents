"""Shared invocation contract for Hill-climbing, SA, and GA implementations."""

from collections.abc import Callable
from typing import Protocol, TypeAlias, TypeVar

from src.results import ObjectiveValue, ParameterValue, SearchResult


StateT = TypeVar("StateT")
ObjectiveFunction: TypeAlias = Callable[[StateT], ObjectiveValue]


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
    return lambda candidate: candidate.objective()
