"""Parent selection operators."""

from collections.abc import Sequence
from math import isfinite
from random import Random

from src.model import State


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
