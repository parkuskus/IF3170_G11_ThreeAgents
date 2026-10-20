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
from src.model import ORIENTATIONS, FleetState, Placement, State
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
    """Sample an assignment swap or a one-package fleet mutation."""
    if len(state.packages) > 1 and rng.randrange(4) == 0:
        return _fleet_swap_candidate(state, rng)
    return mutate_fleet_individual(state, rng)


def _fleet_swap_candidate(state: FleetState, rng: Random) -> FleetState:
    first, second = rng.sample(state.packages, 2)
    assignments = [
        next(
            ((index, placements[package.id]) for index, placements in enumerate(state.placements)
             if placements[package.id] is not None),
            None,
        )
        for package in (first, second)
    ]
    if assignments == [None, None]:
        return state

    choices = []
    for assignment in assignments:
        if assignment is None:
            orientations = list(ORIENTATIONS)
            rng.shuffle(orientations)
            choices.append(orientations)
        else:
            choices.append([assignment[1].orientation])

    for first_orientation in choices[0]:
        for second_orientation in choices[1]:
            changed = [dict(placements) for placements in state.placements]
            for package, source, target, orientation in (
                (first, assignments[0], assignments[1], first_orientation),
                (second, assignments[1], assignments[0], second_orientation),
            ):
                if source is not None:
                    changed[source[0]][package.id] = None
                if target is not None:
                    changed[target[0]][package.id] = Placement(
                        target[1].coordinate,
                        package.oriented_dimensions(orientation),
                        orientation,
                    )
            candidate = FleetState(state.packages, state.trucks, tuple(changed))
            if candidate != state and not validate_fleet_state(candidate):
                return candidate
    return state
