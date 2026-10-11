import unittest
from random import Random

from src.algorithms.genetic_algorithm import (
    initialize_population,
    mutate_individual,
    roulette_wheel_select,
    single_point_crossover,
)
from src.model import Package, Placement, State, Truck
from src.validation import validate_state


class FixedPointRandom(Random):
    def randrange(self, start, stop=None, step=1):
        return 1


class RecordingRandom(Random):
    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        self.recorded_weights = weights
        return [population[-1]]


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

    def test_roulette_wheel_uses_fitness_as_weights(self):
        population = initialize_population(self.initial_state, 3, Random(3))
        rng = RecordingRandom(1)
        selected = roulette_wheel_select(population, (24, 23, 20), rng)
        self.assertEqual(rng.recorded_weights, [24, 23, 20])
        self.assertIs(selected, population[-1])

    def test_roulette_wheel_shifts_negative_fitness(self):
        population = initialize_population(self.initial_state, 2, Random(3))
        rng = RecordingRandom(1)
        roulette_wheel_select(population, (-3, -1), rng)
        self.assertEqual(rng.recorded_weights, [0, 2])

    def test_single_point_crossover_inherits_parent_prefix_and_suffix(self):
        packages = tuple(
            Package(package_id, (1, 1, 1), 10, 1, False, 1)
            for package_id in ("a", "b", "c")
        )
        truck = Truck((8, 3, 3), 10)
        parent_a = State(
            packages,
            truck,
            {
                "a": Placement((0, 0, 0), (1, 1, 1)),
                "b": Placement((1, 0, 0), (1, 1, 1)),
                "c": Placement((2, 0, 0), (1, 1, 1)),
            },
        )
        parent_b = State(
            packages,
            truck,
            {
                "a": Placement((4, 0, 0), (1, 1, 1)),
                "b": Placement((5, 0, 0), (1, 1, 1)),
                "c": Placement((6, 0, 0), (1, 1, 1)),
            },
        )

        child = single_point_crossover(parent_a, parent_b, FixedPointRandom(1))

        self.assertEqual(child.placements["a"].coordinate, (0, 0, 0))
        self.assertEqual(child.placements["b"].coordinate, (5, 0, 0))
        self.assertEqual(child.placements["c"].coordinate, (6, 0, 0))
        self.assertEqual(validate_state(child), [])

    def test_mutation_changes_at_most_one_gene_and_keeps_state_valid(self):
        mutated = mutate_individual(self.initial_state, Random(41))
        changed_genes = sum(
            self.initial_state.placements[package.id] != mutated.placements[package.id]
            for package in self.initial_state.packages
        )
        self.assertLessEqual(changed_genes, 1)
        self.assertEqual(validate_state(mutated), [])


if __name__ == "__main__":
    unittest.main()
