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
            {"stuck_patience": 0},
        )
        for overrides in invalid_parameters:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_algorithm(**overrides)

    def test_records_current_history_and_temperature_schedule(self):
        result = self.run_algorithm()

        self.assertEqual(len(result.metrics.current_objective_history), 9)
        self.assertEqual(result.metrics.current_objective_history[0], 10)
        self.assertEqual(len(result.metrics.temperature_history), 8)
        self.assertAlmostEqual(result.metrics.temperature_history[1], 8.0)
        for current, best in zip(
            result.metrics.current_objective_history, result.objective_history
        ):
            self.assertLessEqual(current, best)

    def test_acceptance_probabilities_stay_in_unit_interval(self):
        result = self.run_algorithm(max_iterations=60)

        for probability in result.metrics.exp_delta_e_over_temperature:
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_survives_temperature_cooling_to_zero(self):
        # 0.5^2000 underflows to 0.0; improving moves must not overflow exp().
        result = self.run_algorithm(
            max_iterations=2000, initial_temperature=1.0, cooling_rate=0.5
        )

        self.assertEqual(result.metrics.temperature_history[-1], 0.0)
        self.assertEqual(validate_state(result.final_state), [])

    def test_counts_each_stalled_plateau_once(self):
        # A constant objective never improves, so the run is a single plateau.
        result = self.run_algorithm(
            max_iterations=50, stuck_patience=5, objective_fn=lambda state: 0
        )
        self.assertEqual(result.metrics.stuck_count, 1)
        self.assertEqual(result.metrics.stuck_patience, 5)

        never_stuck = self.run_algorithm(
            max_iterations=4, stuck_patience=5, objective_fn=lambda state: 0
        )
        self.assertEqual(never_stuck.metrics.stuck_count, 0)

    def test_new_plateau_after_improvement_is_counted_again(self):
        calls = iter(range(10_000))
        # Objective improves only on the 11th evaluation, splitting the run
        # into two plateaus that each exceed the patience.
        result = self.run_algorithm(
            max_iterations=30,
            stuck_patience=5,
            objective_fn=lambda state: 1 if next(calls) >= 10 else 0,
        )
        self.assertEqual(result.metrics.stuck_count, 2)


if __name__ == "__main__":
    unittest.main()