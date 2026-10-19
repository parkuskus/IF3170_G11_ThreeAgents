"Simulated Annealing Algorithm"

from math import exp, isfinite
from random import Random
from time import perf_counter

from src.contracts import ObjectiveFunction, resolve_objective
from src.model import State
from src.neighbors import random_neighbor
from src.results import SearchResult, SimulatedAnnealingMetrics


def simulated_annealing(
    initial_state: State,
    *,
    seed: int,
    objective_fn: ObjectiveFunction[State] | None = None,
    max_iterations: int,
    initial_temperature: float,
    cooling_rate: float,
    objective_name: str | None = None,
    run_index: int = 1,
    stuck_patience: int = 100,
) -> SearchResult[State]:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("max_iterations must be a non-negative integer")
    if (
        isinstance(initial_temperature, bool)
        or not isinstance(initial_temperature, (int, float))
        or not isfinite(initial_temperature)
        or initial_temperature <= 0
    ):
        raise ValueError("initial_temperature must be a finite positive number")
    if (
        isinstance(cooling_rate, bool)
        or not isinstance(cooling_rate, (int, float))
        or not isfinite(cooling_rate)
        or not 0 < cooling_rate <= 1
    ):
        raise ValueError("cooling_rate must be a finite number between 0 and 1")
    if objective_name is not None and (
        not isinstance(objective_name, str) or not objective_name
    ):
        raise ValueError("objective_name must be a non-empty string")
    if type(run_index) is not int or run_index < 1:
        raise ValueError("run_index must be a positive integer")
    if type(stuck_patience) is not int or stuck_patience < 1:
        raise ValueError("stuck_patience must be a positive integer")

    started_at = perf_counter()
    rng = Random(seed)
    objective = resolve_objective(initial_state, objective_fn)
    current = initial_state
    best = current
    current_score = _validate_objective(objective(current))
    best_score = current_score
    objective_history = [best_score]
    current_history = [current_score]
    acceptance_history = []
    temperature_history = []
    stuck_count = 0
    stalled_iterations = 0
    plateau_counted = False
    temperature = float(initial_temperature)

    for _ in range(max_iterations):
        candidate = random_neighbor(current, rng)
        candidate_score = _validate_objective(objective(candidate))
        delta = candidate_score - current_score
        acceptance_probability = _acceptance_probability(delta, temperature)
        acceptance_history.append(acceptance_probability)
        temperature_history.append(temperature)

        if delta >= 0 or rng.random() < acceptance_probability:
            current = candidate
            current_score = candidate_score

        if current_score > best_score:
            best = current
            best_score = current_score
            stalled_iterations = 0
            plateau_counted = False
        else:
            stalled_iterations += 1
            if stalled_iterations >= stuck_patience and not plateau_counted:
                stuck_count += 1
                plateau_counted = True
        objective_history.append(best_score)
        current_history.append(current_score)
        temperature *= cooling_rate

    return SearchResult(
        algorithm="simulated_annealing",
        objective_name=objective_name or ("total_value" if objective_fn is None else "custom"),
        initial_state=initial_state,
        final_state=best,
        objective_value=best_score,
        objective_history=tuple(objective_history),
        duration_seconds=perf_counter() - started_at,
        metrics=SimulatedAnnealingMetrics(
            iterations=max_iterations,
            exp_delta_e_over_temperature=tuple(acceptance_history),
            stuck_count=stuck_count,
            stuck_patience=stuck_patience,
            current_objective_history=tuple(current_history),
            temperature_history=tuple(temperature_history),
        ),
        seed=seed,
        run_index=run_index,
        parameters={
            "max_iterations": max_iterations,
            "initial_temperature": initial_temperature,
            "cooling_rate": cooling_rate,
            "stuck_patience": stuck_patience,
        },
    )


def _acceptance_probability(delta: float, temperature: float) -> float:
    if delta >= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return exp(delta / temperature)


def _validate_objective(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(
            "objective_fn must return finite integer or float values"
        )
    return value