"""Generational Genetic Algorithm loop."""

from math import isfinite
from random import Random
from time import perf_counter
from typing import TypeVar

from src.algorithms.genetic_algorithm.selection import roulette_wheel_select
from src.contracts import ObjectiveFunction, SearchProblem
from src.results import GeneticAlgorithmMetrics, ObjectiveValue, SearchResult


StateT = TypeVar("StateT")


def _evaluate_population(
    population: tuple[StateT, ...], objective_fn: ObjectiveFunction[StateT]
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
    initial_state: StateT,
    *,
    seed: int,
    objective_fn: ObjectiveFunction[StateT] | None = None,
    search_problem: SearchProblem[StateT] | None = None,
    population_size: int,
    generations: int,
    mutation_rate: float = 0.05,
    selection_mode: str = "roulette",
    crossover_enabled: bool = True,
    objective_name: str | None = None,
    run_index: int = 1,
) -> SearchResult[StateT]:
    """Run GA and return the best-so-far state plus per-generation metrics."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(generations) is not int or generations < 0:
        raise ValueError("generations must be a non-negative integer")
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not isfinite(mutation_rate)
        or not 0 <= mutation_rate <= 1
    ):
        raise ValueError("mutation_rate must be a finite number between 0 and 1")
    if objective_name is not None and (not isinstance(objective_name, str) or not objective_name):
        raise ValueError("objective_name must be a non-empty string")
    if selection_mode not in ("roulette", "uniform"):
        raise ValueError("selection_mode must be 'roulette' or 'uniform'")
    if type(crossover_enabled) is not bool:
        raise ValueError("crossover_enabled must be a boolean")
    if type(run_index) is not int or run_index < 1:
        raise ValueError("run_index must be a positive integer")

    started_at = perf_counter()
    rng = Random(seed)
    from src.search_problems import search_problem_for_state

    problem = search_problem or search_problem_for_state(initial_state, objective_fn)
    objective = objective_fn if objective_fn is not None else problem.objective_fn
    if not problem.is_valid(initial_state):
        raise ValueError("initial_state is invalid")
    if generations > 0 and crossover_enabled and problem.crossover is None:
        raise ValueError("search_problem must provide crossover when crossover is enabled")
    if generations > 0 and mutation_rate > 0 and problem.mutate is None:
        raise ValueError("search_problem must provide mutate when mutation_rate is positive")
    population = (initial_state,) + tuple(
        problem.random_state(rng) for _ in range(population_size - 1)
    )
    if any(not problem.is_valid(individual) for individual in population):
        raise ValueError("initial-state operator returned invalid state")
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
            if selection_mode == "uniform":
                parent_a = rng.choice(population)
                parent_b = rng.choice(population)
            else:
                parent_a = roulette_wheel_select(population, fitnesses, rng)
                parent_b = roulette_wheel_select(population, fitnesses, rng)
            child = (
                problem.crossover(parent_a, parent_b, rng)
                if crossover_enabled
                else parent_a
            )
            if not problem.is_valid(child):
                raise ValueError("crossover operator returned invalid state")
            if rng.random() < mutation_rate:
                assert problem.mutate is not None
                child = problem.mutate(child, rng)
                if not problem.is_valid(child):
                    raise ValueError("mutation operator returned invalid state")
            next_population.append(child)
        population = tuple(next_population)

    assert best_score is not None
    return SearchResult(
        algorithm="genetic_algorithm",
        objective_name=objective_name or (
            "custom" if objective_fn is not None else problem.objective_name
        ),
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
            "selection_mode": selection_mode,
            "crossover_enabled": crossover_enabled,
        },
    )
