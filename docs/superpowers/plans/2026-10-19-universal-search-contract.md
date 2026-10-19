# Universal Search Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Hill Climbing, Simulated Annealing, and Genetic Algorithm search any state type through one reusable, typed problem contract.

**Architecture:** Add a generic `SearchProblem[StateT]` carrying objective, validity, initial-state, and neighbor operations, with optional crossover and mutation operations for GA. Provide adapters for existing single-truck states and fleet states; make each algorithm consume this contract while preserving its current call form and defaults. Replace the separate fleet GA loop with the generic GA plus the fleet adapter.

**Tech Stack:** Python 3.10+, standard library, `unittest`.

**Spec:** User request: “bikin implementasi multi trucknya itu modular sehingga bisa dipakai di berbagai jenis algoritma ... buat kontraknya universal.”

## Global Constraints

- Preserve existing single-truck callers and default results.
- All algorithms operate on generic state types and use the same objective/validity/initial/neighbor contract.
- GA-only crossover and mutation hooks remain optional for algorithms that do not use them.
- Keep fleet-specific geometry/placement logic in fleet adapters, not algorithm loops.
- Do not add dependencies.

## Review Focus

- Existing single-truck entry points without a contract: keep behavior via default adapters.
- Custom state type without `objective()`: support it when supplied an explicit problem/objective.
- Invalid initial states and invalid operator-produced states: reject them at the algorithm boundary.
- GA problem missing crossover/mutation hooks: report a clear error only when that operator is requested.
- Determinism and existing metrics/results: preserve seeded behavior and run the existing suite.

---

### Task 1: Define Universal Problem Contract and Domain Adapters

**Files:**
- Modify: `src/contracts.py`
- Create: `src/search_problems.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- `SearchProblem[StateT]` contains `objective_fn`, `is_valid`, `random_state`, `neighbor`, and optional `crossover`/`mutate` callables.
- `search_problem_for_state(state, objective_fn=None) -> SearchProblem` returns the existing single-truck or fleet adapter.

- [x] Test generic custom-state contracts and default adapters for `State` and `FleetState`.
- [x] Run focused tests and confirm missing contract/adapters fail.
- [x] Implement typed protocol/dataclass and adapters using existing validators and operators.
- [x] Run contract tests.

### Task 2: Generalize Hill Climbing and Simulated Annealing

**Files:**
- Modify: `src/algorithms/hill_climbing/algorithm.py`
- Modify: `src/algorithms/simulated_annealing/algorithm.py`
- Test: `tests/test_hill_climbing.py`
- Test: `tests/test_simulated_annealing.py`

**Interfaces:**
- Both algorithms accept optional `search_problem: SearchProblem[StateT]` and return `SearchResult[StateT]`.
- Existing callers may continue to pass `initial_state` and `objective_fn` without a problem.

- [x] Add tests that run HC (including random restart) and SA on `FleetState` via the shared contract, and reject invalid states from an operator.
- [x] Run focused tests and observe the expected unsupported-state failures.
- [x] Generalize internal state types and route objective, validity, neighbor, and restart initialization through the contract.
- [x] Run focused algorithm tests.

### Task 3: Generalize GA and Reuse It for Fleet Search

**Files:**
- Modify: `src/algorithms/genetic_algorithm/algorithm.py`
- Modify: `src/algorithms/genetic_algorithm/fleet.py`
- Modify: `src/algorithms/genetic_algorithm/selection.py`
- Test: `tests/test_genetic_algorithm.py`

**Interfaces:**
- `genetic_algorithm(..., search_problem: SearchProblem[StateT] | None = None) -> SearchResult[StateT]` uses contract-provided initial states, crossover, mutation, objective, and validation.
- `fleet_genetic_algorithm` remains a compatibility wrapper that invokes generic GA with the fleet problem adapter.

- [x] Test generic GA on a custom state, missing GA operators, fleet adapter reuse, and valid final state.
- [x] Run focused tests and confirm the generic API/hook errors fail as expected.
- [x] Generalize the loop and remove the duplicate fleet GA evolution loop while retaining its public wrapper.
- [x] Run all GA tests.

### Task 4: Public Exports and Documentation

**Files:**
- Modify: `src/__init__.py` (only if needed)
- Modify: `README.md`
- Test: full `tests/` suite

- [x] Document the shared contract and show that single-truck and fleet problems can be passed to each algorithm.
- [x] Run `python -m unittest discover -s tests -v` and `git diff --check`.
