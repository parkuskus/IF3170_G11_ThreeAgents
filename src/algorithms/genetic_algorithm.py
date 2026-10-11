"""Initial population utilities for the Genetic Algorithm."""

from random import Random

from src.model import State
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
