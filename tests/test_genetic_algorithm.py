import unittest
from random import Random

from src.algorithms.genetic_algorithm import initialize_population
from src.model import Package, Placement, State, Truck
from src.validation import validate_state


class GeneticAlgorithmFoundationTests(unittest.TestCase):
    def setUp(self):
        packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (1, 1, 1), 20, 1, False, 1),
        )
        self.initial_state = State(
            packages,
            Truck((3, 3, 3), 5),
            {"a": Placement((0, 0, 0), (1, 1, 1)), "b": None},
        )

    def test_population_has_requested_size_and_preserves_initial_state(self):
        population = initialize_population(self.initial_state, 6, Random(7))
        self.assertEqual(len(population), 6)
        self.assertIs(population[0], self.initial_state)
        self.assertTrue(all(not validate_state(state) for state in population))

    def test_population_is_reproducible_for_same_seed(self):
        first = initialize_population(self.initial_state, 5, Random(17))
        second = initialize_population(self.initial_state, 5, Random(17))
        self.assertEqual(first, second)

    def test_invalid_population_size_is_rejected(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                initialize_population(self.initial_state, size, Random(1))

    def test_invalid_initial_state_is_rejected(self):
        invalid = State(
            self.initial_state.packages,
            self.initial_state.truck,
            {
                "a": Placement((3, 0, 0), (1, 1, 1)),
                "b": None,
            },
        )
        with self.assertRaisesRegex(ValueError, "initial_state is invalid"):
            initialize_population(invalid, 2, Random(1))

    def test_population_size_one_contains_only_initial_state(self):
        population = initialize_population(self.initial_state, 1, Random(1))
        self.assertEqual(population, (self.initial_state,))


if __name__ == "__main__":
    unittest.main()
