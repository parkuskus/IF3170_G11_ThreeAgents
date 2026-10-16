import unittest
from random import Random

from src.algorithms.genetic_algorithm import (
    genetic_algorithm,
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

    def test_roulette_wheel_falls_back_to_uniform_for_zero_fitness(self):
        population = initialize_population(self.initial_state, 2, Random(3))
        self.assertIn(
            roulette_wheel_select(population, (0, 0), Random(4)), population
        )

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

    def test_single_point_crossover_repairs_conflicting_inherited_placements(self):
        packages = (
            Package("base", (2, 2, 1), 10, 1, False, 1),
            Package("top", (1, 1, 1), 10, 1, False, 1),
            Package("side", (1, 1, 1), 10, 1, False, 1),
        )
        truck = Truck((6, 4, 3), 10)
        parent_a = State(
            packages,
            truck,
            {
                "base": Placement((0, 0, 0), (2, 2, 1)),
                "top": Placement((0, 0, 1), (1, 1, 1)),
                "side": Placement((3, 0, 0), (1, 1, 1)),
            },
        )
        parent_b = State(
            packages,
            truck,
            {
                "base": Placement((4, 0, 0), (2, 2, 1)),
                "top": Placement((0, 0, 0), (1, 1, 1)),
                "side": Placement((2, 0, 0), (1, 1, 1)),
            },
        )
        child = single_point_crossover(parent_a, parent_b, FixedPointRandom(1))
        self.assertEqual(validate_state(child), [])
        self.assertIsNotNone(child.placements["base"])

    def test_mutation_changes_at_most_one_gene_and_keeps_state_valid(self):
        mutated = mutate_individual(self.initial_state, Random(41))
        changed_genes = sum(
            self.initial_state.placements[package.id] != mutated.placements[package.id]
            for package in self.initial_state.packages
        )
        self.assertLessEqual(changed_genes, 1)
        self.assertEqual(validate_state(mutated), [])

    def test_loop_returns_search_result_and_population_histories(self):
        result = genetic_algorithm(
            self.initial_state,
            seed=11,
            population_size=5,
            generations=4,
        )
        self.assertIs(result.initial_state, self.initial_state)
        self.assertEqual(len(result.objective_history), 5)
        self.assertEqual(len(result.metrics.best_objective_history), 5)
        self.assertEqual(len(result.metrics.average_objective_history), 5)
        self.assertEqual(result.objective_history, result.metrics.best_objective_history)
        self.assertEqual(result.objective_value, result.final_state.objective())
        self.assertEqual(validate_state(result.final_state), [])
        self.assertEqual(result.seed, 11)
        self.assertEqual(result.parameters["population_size"], 5)
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_loop_is_reproducible_for_same_seed(self):
        first = genetic_algorithm(
            self.initial_state, seed=23, population_size=6, generations=5
        )
        second = genetic_algorithm(
            self.initial_state, seed=23, population_size=6, generations=5
        )
        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.objective_history, second.objective_history)
        self.assertEqual(first.metrics.average_objective_history, second.metrics.average_objective_history)

    def test_loop_supports_custom_objective(self):
        result = genetic_algorithm(
            self.initial_state,
            seed=5,
            population_size=3,
            generations=1,
            objective_fn=lambda state: -state.objective(),
            objective_name="negative_value",
        )
        self.assertEqual(result.objective_name, "negative_value")
        self.assertEqual(result.objective_value, -result.final_state.objective())

    def test_loop_supports_uniform_selection_and_disabled_crossover(self):
        result = genetic_algorithm(
            self.initial_state,
            seed=5,
            population_size=3,
            generations=1,
            selection_mode="uniform",
            crossover_enabled=False,
        )
        self.assertEqual(result.parameters["selection_mode"], "uniform")
        self.assertFalse(result.parameters["crossover_enabled"])
        self.assertEqual(validate_state(result.final_state), [])

    def test_loop_rejects_unknown_ablation_operator_settings(self):
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            genetic_algorithm(
                self.initial_state, seed=1, population_size=2, generations=1,
                selection_mode="tournament",
            )

    def test_loop_rejects_invalid_run_parameters(self):
        invalid_parameters = (
            {"seed": True, "population_size": 3, "generations": 1},
            {"seed": 1, "population_size": 3, "generations": -1},
            {"seed": 1, "population_size": 3, "generations": 1, "mutation_rate": 1.1},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                genetic_algorithm(self.initial_state, **parameters)


if __name__ == "__main__":
    unittest.main()
