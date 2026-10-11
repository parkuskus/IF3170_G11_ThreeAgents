"""Genetic Algorithm population and genetic operators."""

from collections.abc import Sequence
from math import isfinite
from random import Random

from src.model import ORIENTATIONS, Package, Placement, State
from src.neighbors import random_initial_state
from src.validation import validate_state


def initialize_population(
    initial_state: State, population_size: int, rng: Random
) -> tuple[State, ...]:
    """Create a valid population with the random initial state as member 0."""
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    violations = validate_state(initial_state)
    if violations:
        raise ValueError(f"initial_state is invalid: {'; '.join(violations)}")

    population = [initial_state]
    population.extend(
        random_initial_state(initial_state.packages, initial_state.truck, rng)
        for _ in range(population_size - 1)
    )
    return tuple(population)


def roulette_wheel_select(
    population: Sequence[State], fitnesses: Sequence[float], rng: Random
) -> State:
    """Select one parent with probability proportional to its fitness."""
    if not population or len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same non-zero length")
    if any(not isfinite(fitness) for fitness in fitnesses):
        raise ValueError("fitness values must be finite numbers")

    minimum = min(fitnesses)
    weights = [fitness - minimum if minimum < 0 else fitness for fitness in fitnesses]
    if sum(weights) == 0:
        return rng.choice(population)
    return rng.choices(population, weights=weights, k=1)[0]


def _check_compatible_parents(parent_a: State, parent_b: State) -> None:
    if parent_a.packages != parent_b.packages or parent_a.truck != parent_b.truck:
        raise ValueError("parents must describe the same packages and truck")
    if validate_state(parent_a) or validate_state(parent_b):
        raise ValueError("parents must be valid states")


def single_point_crossover(parent_a: State, parent_b: State, rng: Random) -> State:
    """Create one valid child by crossing package genes at one random point."""
    _check_compatible_parents(parent_a, parent_b)
    packages = parent_a.packages
    if not packages:
        return parent_a

    cut = rng.randrange(1, len(packages)) if len(packages) > 1 else 1
    inherited = {
        package.id: (
            parent_a.placements[package.id]
            if index < cut
            else parent_b.placements[package.id]
        )
        for index, package in enumerate(packages)
    }
    empty = {package.id: None for package in packages}
    child = State(packages, parent_a.truck, empty)

    # Place inherited genes from low to high so a supporting package is tried first.
    placed_genes = sorted(
        (
            (package, placement)
            for package in packages
            if (placement := inherited[package.id]) is not None
        ),
        key=lambda item: (item[1].coordinate[2], packages.index(item[0])),
    )
    for package, placement in placed_genes:
        placements = dict(child.placements)
        placements[package.id] = placement
        candidate = State(packages, parent_a.truck, placements)
        if not validate_state(candidate):
            child = candidate
        else:
            repaired = _random_valid_placement(child, package, rng)
            if repaired is not None:
                placements = dict(child.placements)
                placements[package.id] = repaired
                child = State(packages, parent_a.truck, placements)

    return child


def _random_valid_placement(
    state: State,
    package: Package,
    rng: Random,
    *,
    orientation: tuple[int, int, int] | None = None,
    coordinate: tuple[int, int, int] | None = None,
    attempts: int = 50,
) -> Placement | None:
    """Find a random placement for one gene that keeps the whole state valid."""
    orientations = (orientation,) if orientation is not None else ORIENTATIONS
    others = {
        package_id: placement
        for package_id, placement in state.placements.items()
        if package_id != package.id and placement is not None
    }
    packages_by_id = {item.id: item for item in state.packages}
    possible_z = {0}
    possible_z.update(
        placement.coordinate[2] + placement.oriented_dimensions[2]
        for package_id, placement in others.items()
        if not packages_by_id[package_id].fragile
    )

    for _ in range(attempts):
        selected_orientation = rng.choice(orientations)
        dimensions = package.oriented_dimensions(selected_orientation)
        if any(size > limit for size, limit in zip(dimensions, state.truck.dimensions)):
            continue
        if coordinate is None:
            z_values = [
                z for z in possible_z if z + dimensions[2] <= state.truck.dimensions[2]
            ]
            if not z_values:
                continue
            position = (
                rng.randint(0, state.truck.dimensions[0] - dimensions[0]),
                rng.randint(0, state.truck.dimensions[1] - dimensions[1]),
                rng.choice(z_values),
            )
        else:
            position = coordinate
        placement = Placement(position, dimensions, selected_orientation)
        placements = dict(state.placements)
        placements[package.id] = placement
        if not validate_state(State(state.packages, state.truck, placements)):
            return placement
    return None


def mutate_individual(
    individual: State, rng: Random, attempts: int = 50
) -> State:
    """Mutate one package gene; return the original if no valid mutation exists."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if validate_state(individual):
        raise ValueError("individual must be a valid state")
    if not individual.packages:
        return individual

    package = rng.choice(individual.packages)
    old = individual.placements[package.id]
    operations = ("insert",) if old is None else ("move", "rotate", "remove")
    for _ in range(attempts):
        operation = rng.choice(operations)
        if operation == "remove":
            placement = None
        elif operation == "rotate":
            assert old is not None
            orientations = tuple(o for o in ORIENTATIONS if o != old.orientation)
            placement = _random_valid_placement(
                individual,
                package,
                rng,
                orientation=rng.choice(orientations),
                coordinate=old.coordinate,
                attempts=1,
            )
        else:
            placement = _random_valid_placement(
                individual,
                package,
                rng,
                orientation=old.orientation if old is not None else None,
                attempts=1,
            )
        if placement == old:
            continue
        placements = dict(individual.placements)
        placements[package.id] = placement
        candidate = State(individual.packages, individual.truck, placements)
        if not validate_state(candidate):
            return candidate
    return individual
