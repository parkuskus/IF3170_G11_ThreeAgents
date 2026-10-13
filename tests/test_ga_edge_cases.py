import unittest
from random import Random
from pathlib import Path

from src.algorithms.genetic_algorithm import (
    genetic_algorithm,
    initialize_population,
    single_point_crossover,
)
from src.experiments.data import load_problem
from src.model import Package, Placement, State, Truck
from src.neighbors import random_initial_state
from src.validation import validate_state


class GeneticAlgorithmEdgeCaseTests(unittest.TestCase):
    def test_complex_fixture_population_and_search_stay_valid(self):
        problem = Path(__file__).parent / "fixtures" / "ga_hard_problem.json"
        packages, truck = load_problem(problem)
        initial = random_initial_state(packages, truck, Random(103), attempts_per_package=100)
        population = initialize_population(initial, 8, Random(104))
        self.assertTrue(all(validate_state(state) == [] for state in population))

        result = genetic_algorithm(
            initial,
            seed=105,
            population_size=8,
            generations=5,
        )
        self.assertEqual(validate_state(result.final_state), [])
        self.assertEqual(len(result.metrics.best_objective_history), 6)
        self.assertEqual(len(result.metrics.average_objective_history), 6)
        self.assertLessEqual(result.objective_value, sum(package.value for package in packages))

    def test_orientation_allows_package_that_only_fits_after_rotation(self):
        package = Package("rotate", (4, 2, 3), 10, 1, False, 1)
        truck = Truck((2, 3, 4), 1)
        state = random_initial_state((package,), truck, Random(7), attempts_per_package=100)
        self.assertIsNotNone(state.placements["rotate"])
        self.assertEqual(state.placements["rotate"].oriented_dimensions, (2, 3, 4))

    def test_infeasible_packages_remain_outside_and_zero_fitness_ga_runs(self):
        packages = (Package("oversized", (7, 1, 1), 0, 0, False, 1),)
        truck = Truck((6, 6, 6), 0)
        initial = random_initial_state(packages, truck, Random(8))
        self.assertIsNone(initial.placements["oversized"])
        result = genetic_algorithm(
            initial, seed=9, population_size=3, generations=2
        )
        self.assertEqual(result.objective_value, 0)
        self.assertEqual(validate_state(result.final_state), [])

    def test_placement_exactly_touching_truck_walls_is_valid(self):
        package = Package("edge", (2, 2, 2), 1, 1, False, 1)
        state = State(
            (package,),
            Truck((4, 4, 2), 1),
            {"edge": Placement((2, 2, 0), (2, 2, 2))},
        )
        self.assertEqual(validate_state(state), [])

    def test_crossover_rejects_parents_from_different_problems(self):
        a = Package("a", (1, 1, 1), 1, 1, False, 1)
        b = Package("b", (1, 1, 1), 1, 1, False, 1)
        truck = Truck((2, 2, 2), 2)
        parent_a = State((a,), truck, {"a": None})
        parent_b = State((b,), truck, {"b": None})
        with self.assertRaisesRegex(ValueError, "same packages and truck"):
            single_point_crossover(parent_a, parent_b, Random(1))


if __name__ == "__main__":
    unittest.main()
