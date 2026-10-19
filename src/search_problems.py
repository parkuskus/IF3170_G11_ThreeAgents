"""Adapters from built-in state types to the universal search contract."""

from random import Random

from src.algorithms.genetic_algorithm.crossover import single_point_crossover
from src.algorithms.genetic_algorithm.fleet import (
    fleet_single_point_crossover,
    mutate_fleet_individual,
    random_initial_fleet_state,
)
from src.algorithms.genetic_algorithm.mutation import mutate_individual
from src.contracts import ObjectiveFunction, SearchProblem, resolve_objective
from src.model import FleetState, State
from src.neighbors import random_initial_state, random_neighbor
from src.validation import validate_fleet_state, validate_state


def search_problem_for_state(
    state: State | FleetState, objective_fn: ObjectiveFunction | None = None
) -> SearchProblem:
    if isinstance(state, FleetState):
        objective = resolve_objective(state, objective_fn)
        return SearchProblem(
            objective,
            lambda candidate: not validate_fleet_state(candidate),
            lambda rng: random_initial_fleet_state(state.packages, state.trucks, rng),
            lambda candidate, rng: _fleet_neighbor(candidate, rng),
            fleet_single_point_crossover,
            mutate_fleet_individual,
            "custom" if objective_fn is not None else "total_value",
        )
    if isinstance(state, State):
        objective = resolve_objective(state, objective_fn)
        return SearchProblem(
            objective,
            lambda candidate: not validate_state(candidate),
            lambda rng: random_initial_state(state.packages, state.truck, rng),
            random_neighbor,
            single_point_crossover,
            mutate_individual,
            "custom" if objective_fn is not None else "total_value",
        )
    raise TypeError("a SearchProblem is required for this state type")


def _fleet_neighbor(state: FleetState, rng: Random) -> FleetState:
    """Use fleet mutation as the random one-step neighbor operation."""
    return mutate_fleet_individual(state, rng)
