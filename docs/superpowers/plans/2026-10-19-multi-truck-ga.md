# Multi-Truck Genetic Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible GA mode that assigns packages across a configurable fleet of trucks and maximizes total fleet package value.

**Architecture:** Keep the existing single-truck `State`, algorithms, and JSON inputs working unchanged. Add a `FleetState` with one shared package set and one placement map per truck, validate each truck using existing single-truck constraints plus fleet-wide uniqueness, and use fleet-aware GA operators for initialization, crossover, and mutation. Accept a `trucks` JSON list for individually configurable dimensions/capacities, while old `truck` JSON plus a CLI truck-count option continues to create identical copies. A dedicated fleet runner serializes results and visualizes states; infeasible packages remain outside the fleet.

**Tech Stack:** Python 3.10+, standard library, existing `unittest` suite and matplotlib visualization dependencies.

**Spec:** `target/Spesifikasi Tugas Besar 1 IF3170 Inteligensi Artifisial 2026_2027.md`, Bonus > Multi-truk.

## Global Constraints

- Do not change existing single-truck algorithm behavior or input compatibility.
- Every package is loaded in at most one truck; each truck independently obeys geometry, support, fragility, and weight capacity constraints.
- Fleet objective is the sum of values of all packages loaded across the fleet.
- Every configured truck has positive dimensions and non-negative capacity; trucks may differ.
- A positive CLI truck count duplicates the legacy single `truck` definition; it cannot override an explicit `trucks` list.
- Preserve seeded reproducibility and use only the project's current dependencies.

## Review Focus

- Duplicate package placement across trucks: test fleet validation rejects it.
- Over-capacity or invalid geometry in any one truck: test fleet validation rejects the fleet.
- A package that cannot fit any truck: test it remains outside while the rest of the fleet state is valid.
- Crossover conflict in one truck with another truck available: test repair tries another truck and returns a valid child.
- Existing single-truck flows: run the full existing suite unchanged.

---

### Task 1: Fleet State and Validation

**Files:**
- Modify: `src/model.py`
- Modify: `src/validation.py`
- Test: `tests/test_model.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces `FleetState(packages: tuple[Package, ...], trucks: tuple[Truck, ...], placements: tuple[Mapping[str, Placement | None], ...])` with `objective() -> int`.
- Produces `validate_fleet_state(state: FleetState) -> list[str]`.

- [x] Add tests for non-empty fleet/state shape, summed objective, duplicate package assignments, heterogeneous trucks, and per-truck capacity/geometry validation.
- [x] Run the focused tests and confirm expected failures before implementation.
- [x] Implement the minimum FleetState and fleet validator, reusing `validate_state` for each truck projection.
- [x] Run focused tests and the existing model/validation tests.

### Task 2: Fleet GA Operators and Solver

**Files:**
- Create: `src/algorithms/genetic_algorithm/fleet.py`
- Modify: `src/algorithms/genetic_algorithm/__init__.py`
- Test: `tests/test_genetic_algorithm.py`

**Interfaces:**
- Produces `random_initial_fleet_state(packages: tuple[Package, ...], trucks: tuple[Truck, ...], rng: Random) -> FleetState`.
- Produces `fleet_genetic_algorithm(initial_state, *, seed, population_size, generations, mutation_rate=0.05) -> SearchResult[FleetState]`.

- [x] Add tests for seeded valid initialization, aggregate-value scoring, crossover repair/reassignment, mutation across truck assignments, valid final results, and invalid truck count.
- [x] Run focused tests and confirm they fail because the fleet API is absent.
- [x] Implement fleet-aware population, crossover, and mutation using existing placement validation; crossover tries inherited truck first, then other trucks, and leaves the package outside only if all placements fail.
- [x] Run focused tests plus the existing GA tests.

### Task 3: Configurable Fleet Input, CLI, and Result Artifacts

**Files:**
- Create: `src/experiments/run_fleet_ga.py`
- Create: `src/experiments/fleet_genetic_algorithm.py`
- Modify: `src/experiments/data.py`
- Modify: `src/visualization/genetic_algorithm.py`
- Test: `tests/test_fleet_experiments.py`
- Test: `tests/test_ga_data.py`
- Modify: `README.md`

**Interfaces:**
- Input accepts either legacy `truck` or `trucks: [{dimensions, max_capacity}, ...]`; explicit `trucks` values configure each vehicle independently.
- `python -m src.experiments.run_fleet_ga` accepts optional `--truck-count N` only with legacy `truck`; it makes N identical copies.
- Fleet result JSON includes truck count, each truck's loaded packages/placements, and total fleet value.

- [x] Add loader and CLI integration tests for legacy replicated trucks and heterogeneous `trucks` JSON; verify serialized per-truck assignments and total value.
- [x] Run the focused test and confirm the new CLI option is not handled yet.
- [x] Implement a dedicated fleet runner and CLI, fleet result serialization, and a minimal fleet state visualization; document CLI usage and the additional fleet operations.
- [x] Run the focused CLI/experiment tests and the full suite with `python -m unittest discover -s tests -v`.
