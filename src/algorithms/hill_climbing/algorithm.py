from math import isfinite
from random import Random
from time import perf_counter
from typing import TypeVar

from src.contracts import ObjectiveFunction, SearchProblem
from src.results import HillClimbingMetrics, ObjectiveValue, SearchResult

StateT = TypeVar("StateT")
VARIANTS = ("steepest", "sideways", "stochastic", "random_restart")


def _check_int(value, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _score(objective: ObjectiveFunction[StateT], state: StateT) -> ObjectiveValue:
    score = objective(state)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
        raise ValueError("objective_fn harus mengembalikan finite integer atau float values")
    return score


def _climb(
    start: StateT,
    problem: SearchProblem[StateT],
    objective: ObjectiveFunction[StateT],
    rng: Random,
    *,
    variant: str,
    neighbors_per_iteration: int,
    max_iterations: int,
    max_sideways: int,
) -> tuple[StateT, ObjectiveValue, list[ObjectiveValue], int, int]:

    current, current_score = start, _score(objective, start)
    history = [current_score]
    iterations = sideways = 0

    while iterations < max_iterations:
        if variant == "stochastic":
            candidates = [problem.neighbor(current, rng)]
        else:
            candidates = [problem.neighbor(current, rng) for _ in range(neighbors_per_iteration)]
        if any(not problem.is_valid(candidate) for candidate in candidates):
            raise ValueError("neighbor operator returned invalid state")
        scored = [(_score(objective, c), c) for c in candidates]
        best_score, best = max(scored, key=lambda pair: pair[0])

        if best_score > current_score:
            current, current_score = best, best_score
        elif variant == "sideways" and best_score == current_score and sideways < max_sideways:
            current = best
            sideways += 1
        elif variant == "stochastic":
            iterations += 1  
            history.append(current_score)
            continue
        else:
            break  
        iterations += 1
        history.append(current_score)

    return current, current_score, history, iterations, sideways


def hill_climbing(
    initial_state: StateT,
    *,
    seed: int,
    objective_fn: ObjectiveFunction[StateT] | None = None,
    search_problem: SearchProblem[StateT] | None = None,
    variant: str = "steepest",
    neighbors_per_iteration: int = 30,
    max_iterations: int = 1000,
    max_sideways: int = 100,
    max_restarts: int = 10,
    objective_name: str | None = None,
    run_index: int = 1,
) -> SearchResult[StateT]:
    if variant not in VARIANTS:
        raise ValueError(f"variant harus salah satu dari {VARIANTS}")
    if type(seed) is not int:
        raise ValueError("seed harus berupa bilangan bulat")
    _check_int(neighbors_per_iteration, "neighbors_per_iteration", 1)
    _check_int(max_iterations, "max_iterations", 0)
    _check_int(max_sideways, "max_sideways", 0)
    _check_int(max_restarts, "max_restarts", 0)
    _check_int(run_index, "run_index", 1)

    started_at = perf_counter()
    rng = Random(seed)
    from src.search_problems import search_problem_for_state

    problem = search_problem or search_problem_for_state(initial_state, objective_fn)
    objective = objective_fn if objective_fn is not None else problem.objective_fn
    if not problem.is_valid(initial_state):
        raise ValueError("initial_state is invalid")
    climb_kwargs = dict(
        variant="steepest" if variant == "random_restart" else variant,
        neighbors_per_iteration=neighbors_per_iteration,
        max_iterations=max_iterations,
        max_sideways=max_sideways,
    )

    if variant == "random_restart":
        best_state, best_score = initial_state, None
        history: list[ObjectiveValue] = []
        per_restart: list[int] = []
        for restart in range(max_restarts + 1):
            start = (
                initial_state
                if restart == 0
                else problem.random_state(rng)
            )
            if not problem.is_valid(start):
                raise ValueError("initial-state operator returned invalid state")
            state, score, hist, iters, _ = _climb(
                start, problem, objective, rng, **climb_kwargs
            )
            history.extend(hist)
            per_restart.append(iters)
            if best_score is None or score > best_score:
                best_state, best_score = state, score
        final_state, final_score = best_state, best_score
        iterations, sideways = sum(per_restart), 0
        restarts = max_restarts
    else:
        final_state, final_score, history, iterations, sideways = _climb(
            initial_state, problem, objective, rng, **climb_kwargs
        )
        per_restart, restarts = [], 0

    parameters: dict[str, int | float | str | bool] = {
        "variant": variant,
        "max_iterations": max_iterations,
    }
    if variant != "stochastic":
        parameters["neighbors_per_iteration"] = neighbors_per_iteration
    if variant == "sideways":
        parameters["max_sideways"] = max_sideways
    if variant == "random_restart":
        parameters["max_restarts"] = max_restarts

    return SearchResult(
        algorithm=f"hill_climbing_{variant}",
        objective_name=objective_name or (
            "custom" if objective_fn is not None else problem.objective_name
        ),
        initial_state=initial_state,
        final_state=final_state,
        objective_value=final_score,
        objective_history=tuple(history),
        duration_seconds=perf_counter() - started_at,
        metrics=HillClimbingMetrics(
            variant=variant,
            iterations=iterations,
            sideways_moves=sideways,
            maximum_sideways_moves=max_sideways if variant == "sideways" else None,
            restarts=restarts,
            maximum_restarts=max_restarts if variant == "random_restart" else None,
            iterations_per_restart=tuple(per_restart),
        ),
        seed=seed,
        run_index=run_index,
        parameters=parameters,
    )


def steepest_ascent_hill_climbing(initial_state: StateT, **kw) -> SearchResult[StateT]:
    return hill_climbing(initial_state, variant="steepest", **kw)


def sideways_hill_climbing(initial_state: StateT, **kw) -> SearchResult[StateT]:
    return hill_climbing(initial_state, variant="sideways", **kw)


def stochastic_hill_climbing(initial_state: StateT, **kw) -> SearchResult[StateT]:
    return hill_climbing(initial_state, variant="stochastic", **kw)


def random_restart_hill_climbing(initial_state: StateT, **kw) -> SearchResult[StateT]:
    return hill_climbing(initial_state, variant="random_restart", **kw)
