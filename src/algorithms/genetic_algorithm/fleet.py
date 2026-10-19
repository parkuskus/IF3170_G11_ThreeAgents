"""Fleet-aware initialization and genetic operators for the multi-truck bonus."""

from math import isfinite
from random import Random
from time import perf_counter

from src.algorithms.genetic_algorithm._placement import random_valid_placement
from src.algorithms.genetic_algorithm.selection import roulette_wheel_select
from src.contracts import ObjectiveFunction, resolve_objective
from src.model import ORIENTATIONS, FleetState, Package, Placement, State, Truck
from src.results import GeneticAlgorithmMetrics, ObjectiveValue, SearchResult
from src.validation import validate_fleet_state


def _truck_state(fleet: FleetState, index: int) -> State:
    return State(fleet.packages, fleet.trucks[index], fleet.placements[index])


def random_initial_fleet_state(
    packages: tuple[Package, ...], trucks: tuple[Truck, ...], rng: Random
) -> FleetState:
    """Greedily place packages in a seeded random truck order; skip infeasible ones."""
    packages, trucks = tuple(packages), tuple(trucks)
    if not trucks:
        raise ValueError("fleet must contain at least one truck")
    placements = tuple({package.id: None for package in packages} for _ in trucks)
    fleet = FleetState(packages, trucks, placements)
    for package in packages:
        truck_order = list(range(len(trucks)))
        rng.shuffle(truck_order)
        for truck_index in truck_order:
            # shortcut: placement search samples 50 candidates; use exhaustive coordinates if guaranteed packing is required.
            placement = random_valid_placement(
                _truck_state(fleet, truck_index), package, rng
            )
            if placement is None:
                continue
            changed = list(fleet.placements)
            changed[truck_index] = dict(changed[truck_index])
            changed[truck_index][package.id] = placement
            fleet = FleetState(packages, trucks, tuple(changed))
            break
    return fleet


def fleet_single_point_crossover(
    parent_a: FleetState, parent_b: FleetState, rng: Random
) -> FleetState:
    """Cross package truck/placement genes and repair by trying the other trucks."""
    if parent_a.packages != parent_b.packages or parent_a.trucks != parent_b.trucks:
        raise ValueError("parents must describe the same packages and trucks")
    if validate_fleet_state(parent_a) or validate_fleet_state(parent_b):
        raise ValueError("parents must be valid fleet states")
    packages, trucks = parent_a.packages, parent_a.trucks
    if not packages:
        return parent_a

    cut = rng.randrange(1, len(packages)) if len(packages) > 1 else 1
    inherited: dict[str, tuple[int, Placement] | None] = {}
    for package_index, package in enumerate(packages):
        parent = parent_a if package_index < cut else parent_b
        inherited[package.id] = next(
            (
                (truck_index, placements[package.id])
                for truck_index, placements in enumerate(parent.placements)
                if placements[package.id] is not None
            ),
            None,
        )

    fleet = FleetState(
        packages,
        trucks,
        tuple({package.id: None for package in packages} for _ in trucks),
    )
    package_order = {package.id: i for i, package in enumerate(packages)}
    genes = sorted(
        (
            (package, assignment)
            for package in packages
            if (assignment := inherited[package.id]) is not None
        ),
        key=lambda item: (item[1][0], item[1][1].coordinate[2], package_order[item[0].id]),
    )
    for package, (inherited_truck, inherited_placement) in genes:
        truck_order = [inherited_truck] + [
            index for index in range(len(trucks)) if index != inherited_truck
        ]
        for truck_index in truck_order:
            state = _truck_state(fleet, truck_index)
            if truck_index == inherited_truck:
                placement = random_valid_placement(
                    state,
                    package,
                    rng,
                    orientation=inherited_placement.orientation,
                    coordinate=inherited_placement.coordinate,
                    attempts=1,
                )
                if placement is None:
                    placement = random_valid_placement(
                        state, package, rng, orientation=inherited_placement.orientation
                    )
            else:
                placement = random_valid_placement(state, package, rng)
            if placement is None:
                continue
            changed = list(fleet.placements)
            changed[truck_index] = dict(changed[truck_index])
            changed[truck_index][package.id] = placement
            fleet = FleetState(packages, trucks, tuple(changed))
            break
    return fleet


def mutate_fleet_individual(
    individual: FleetState, rng: Random, attempts: int = 50
) -> FleetState:
    """Change one package by moving, rotating, reassigning, or removing it."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if validate_fleet_state(individual):
        raise ValueError("individual must be a valid fleet state")
    if not individual.packages:
        return individual

    package = rng.choice(individual.packages)
    old_truck = next(
        (
            index
            for index, placements in enumerate(individual.placements)
            if placements[package.id] is not None
        ),
        None,
    )
    old = individual.placements[old_truck][package.id] if old_truck is not None else None
    base_maps = [dict(placements) for placements in individual.placements]
    if old_truck is not None:
        base_maps[old_truck][package.id] = None
    base = FleetState(individual.packages, individual.trucks, tuple(base_maps))
    operations = ("insert",) if old is None else ("move", "rotate", "reassign", "remove")

    for _ in range(attempts):
        operation = rng.choice(operations)
        if operation == "remove":
            return base
        if operation == "rotate":
            assert old is not None and old_truck is not None
            placement = random_valid_placement(
                _truck_state(base, old_truck),
                package,
                rng,
                orientation=rng.choice(tuple(o for o in ORIENTATIONS if o != old.orientation)),
                coordinate=old.coordinate,
                attempts=1,
            )
            truck_order = [old_truck]
        else:
            truck_order = list(range(len(base.trucks)))
            if operation == "move" and old_truck is not None:
                truck_order = [old_truck]
            elif operation == "reassign" and old_truck is not None:
                truck_order = [i for i in truck_order if i != old_truck]
            rng.shuffle(truck_order)
            placement = None

        for truck_index in truck_order:
            candidate_placement = placement
            if operation != "rotate":
                candidate_placement = random_valid_placement(
                    _truck_state(base, truck_index),
                    package,
                    rng,
                    orientation=old.orientation if old is not None and operation != "insert" else None,
                    attempts=1,
                )
            if candidate_placement is None:
                continue
            changed = list(base.placements)
            changed[truck_index] = dict(changed[truck_index])
            changed[truck_index][package.id] = candidate_placement
            candidate = FleetState(base.packages, base.trucks, tuple(changed))
            if candidate != individual and not validate_fleet_state(candidate):
                return candidate
    return individual


def fleet_genetic_algorithm(
    initial_state: FleetState,
    *,
    seed: int,
    population_size: int,
    generations: int,
    mutation_rate: float = 0.05,
    objective_fn: ObjectiveFunction[FleetState] | None = None,
    objective_name: str | None = None,
) -> SearchResult[FleetState]:
    """Run the existing GA strategy with fleet-aware individuals and operators."""
    if validate_fleet_state(initial_state):
        raise ValueError("initial_state is invalid")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
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

    started_at = perf_counter()
    rng = Random(seed)
    objective = resolve_objective(initial_state, objective_fn)
    population = (initial_state,) + tuple(
        random_initial_fleet_state(initial_state.packages, initial_state.trucks, rng)
        for _ in range(population_size - 1)
    )
    best_history: list[ObjectiveValue] = []
    average_history: list[float] = []
    best_state = initial_state
    best_score: ObjectiveValue | None = None

    for generation in range(generations + 1):
        scores = tuple(objective(state) for state in population)
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
            for score in scores
        ):
            raise ValueError("objective_fn must return finite integer or float values")
        best_index = max(range(len(population)), key=scores.__getitem__)
        generation_best = scores[best_index]
        best_history.append(generation_best)
        average_history.append(sum(scores) / len(scores))
        if best_score is None or generation_best > best_score:
            best_state, best_score = population[best_index], generation_best
        if generation == generations:
            break

        next_population = []
        while len(next_population) < population_size:
            parent_a = roulette_wheel_select(population, scores, rng)
            parent_b = roulette_wheel_select(population, scores, rng)
            child = fleet_single_point_crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                child = mutate_fleet_individual(child, rng)
            next_population.append(child)
        population = tuple(next_population)

    assert best_score is not None
    return SearchResult(
        algorithm="fleet_genetic_algorithm",
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
        parameters={
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "truck_count": len(initial_state.trucks),
        },
    )
