import unittest

from src.algorithms.simulated_annealing.algorithm import simulated_annealing
from src.model import Package, Placement, State, Truck
from src.validation import validate_state


class SimulatedAnnealingTests(unittest.TestCase):
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

    def run_algorithm(self, **overrides):
        parameters = {
            "seed": 17,
            "max_iterations": 8,
            "initial_temperature": 10.0,
            "cooling_rate": 0.8,
        }
        parameters.update(overrides)
        return simulated_annealing(self.initial_state, **parameters)

    def test_returns_valid_best_state_and_required_metrics(self):
        result = self.run_algorithm()

        self.assertEqual(result.algorithm, "simulated_annealing")
        self.assertEqual(result.initial_state, self.initial_state)
        self.assertEqual(validate_state(result.final_state), [])
        self.assertEqual(result.objective_value, result.final_state.objective())
        self.assertEqual(len(result.objective_history), 9)
        self.assertEqual(result.metrics.iterations, 8)
        self.assertEqual(len(result.metrics.exp_delta_e_over_temperature), 8)
        self.assertEqual(result.parameters["max_iterations"], 8)

    def test_is_reproducible_for_same_seed(self):
        first = self.run_algorithm(seed=23)
        second = self.run_algorithm(seed=23)

        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.objective_history, second.objective_history)
        self.assertEqual(first.metrics, second.metrics)

    def test_supports_custom_objective(self):
        result = self.run_algorithm(
            objective_fn=lambda state: -state.objective(),
            objective_name="negative_value",
        )

        self.assertEqual(result.objective_name, "negative_value")
        self.assertEqual(result.objective_value, -result.final_state.objective())

    def test_rejects_invalid_parameters(self):
        invalid_parameters = (
            {"seed": True},
            {"max_iterations": -1},
            {"initial_temperature": 0},
            {"cooling_rate": 0},
            {"cooling_rate": 1.1},
        )
        for overrides in invalid_parameters:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_algorithm(**overrides)


if __name__ == "__main__":
    unittest.main()