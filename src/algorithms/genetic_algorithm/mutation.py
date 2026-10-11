"""Single-gene mutation operator."""

from random import Random

from src.algorithms.genetic_algorithm._placement import random_valid_placement
from src.model import ORIENTATIONS, State
from src.validation import validate_state


def mutate_individual(individual: State, rng: Random, attempts: int = 50) -> State:
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
            placement = random_valid_placement(
                individual,
                package,
                rng,
                orientation=rng.choice(orientations),
                coordinate=old.coordinate,
                attempts=1,
            )
        else:
            placement = random_valid_placement(
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
