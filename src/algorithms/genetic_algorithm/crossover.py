"""Crossover operators and child validity repair."""

from random import Random

from src.algorithms.genetic_algorithm._placement import random_valid_placement
from src.model import State
from src.validation import validate_state


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
    child = State(
        packages,
        parent_a.truck,
        {package.id: None for package in packages},
    )

    package_order = {package.id: index for index, package in enumerate(packages)}
    placed_genes = sorted(
        (
            (package, placement)
            for package in packages
            if (placement := inherited[package.id]) is not None
        ),
        key=lambda item: (item[1].coordinate[2], package_order[item[0].id]),
    )
    for package, placement in placed_genes:
        placements = dict(child.placements)
        placements[package.id] = placement
        candidate = State(packages, parent_a.truck, placements)
        if not validate_state(candidate):
            child = candidate
            continue

        repaired = random_valid_placement(child, package, rng)
        if repaired is not None:
            placements = dict(child.placements)
            placements[package.id] = repaired
            child = State(packages, parent_a.truck, placements)

    return child
