"""Generational Genetic Algorithm loop."""

from math import isfinite
from random import Random
from time import perf_counter

from src.algorithms.genetic_algorithm.crossover import single_point_crossover
from src.algorithms.genetic_algorithm.mutation import mutate_individual
from src.algorithms.genetic_algorithm.population import initialize_population
from src.algorithms.genetic_algorithm.selection import roulette_wheel_select
from src.contracts import ObjectiveFunction, resolve_objective
from src.model import State
from src.results import GeneticAlgorithmMetrics, ObjectiveValue, SearchResult


def _evaluate_population(
    population: tuple[State, ...], objective_fn: ObjectiveFunction[State]
) -> tuple[ObjectiveValue, ...]:
    scores = tuple(objective_fn(individual) for individual in population)
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(score)
        for score in scores
    ):
        raise ValueError("objective_fn must return finite integer or float values")
    return scores


def genetic_algorithm(
    initial_state: State,
    *,
    seed: int,
    objective_fn: ObjectiveFunction[State] | None = None,
    population_size: int,
    generations: int,
    mutation_rate: float = 0.05,
    objective_name: str | None = None,
    run_index: int = 1,
) -> SearchResult[State]:
    """Run GA and return the best-so-far state plus per-generation metrics."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(generations) is not int or generations < 0:
        raise ValueError("generations must be a non-negative integer")
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not isfinite(mutation_rate)
        or not 0 <= mutation_rate <= 1
    ):
        raise ValueError("mutation_rate must be a finite number between 0 and 1")
    if objective_name is not None and (not isinstance(objective_name, str) or not objective_name):
        raise ValueError("objective_name must be a non-empty string")
    if type(run_index) is not int or run_index < 1:
        raise ValueError("run_index must be a positive integer")

    started_at = perf_counter()
    rng = Random(seed)
    objective = resolve_objective(initial_state, objective_fn)
    population = initialize_population(initial_state, population_size, rng)
    best_history = []
    average_history = []
    best_state = initial_state
    best_score: ObjectiveValue | None = None

    for generation in range(generations + 1):
        fitnesses = _evaluate_population(population, objective)
        generation_best_index = max(range(len(population)), key=fitnesses.__getitem__)
        generation_best_score = fitnesses[generation_best_index]
        best_history.append(generation_best_score)
        average_history.append(sum(fitnesses) / len(fitnesses))

        if best_score is None or generation_best_score > best_score:
            best_state = population[generation_best_index]
            best_score = generation_best_score

        if generation == generations:
            break

        next_population = []
        while len(next_population) < population_size:
            parent_a = roulette_wheel_select(population, fitnesses, rng)
            parent_b = roulette_wheel_select(population, fitnesses, rng)
            child = single_point_crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                child = mutate_individual(child, rng)
            next_population.append(child)
        population = tuple(next_population)

    assert best_score is not None
    return SearchResult(
        algorithm="genetic_algorithm",
        objective_name=objective_name or ("total_value" if objective_fn is None else "custom"),
        initial_state=initial_state,
        final_state=best_state,
        objective_value=best_score,
        objective_history=tuple(best_history),
        duration_seconds=perf_counter() - started_at,
        metrics=GeneticAlgorithmMetrics(
            generations=generations,
            population_size=population_size,
            best_objective_history=tuple(best_history),
            average_objective_history=tuple(average_history),
        ),
        seed=seed,
        run_index=run_index,
        parameters={
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
        },
    )
