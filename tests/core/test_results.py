import unittest

from src.model import Package, Placement, State, Truck
from src.results import (
    GeneticAlgorithmMetrics,
    HillClimbingMetrics,
    SearchResult,
    SimulatedAnnealingMetrics,
)


class SearchResultContractTests(unittest.TestCase):
    def setUp(self):
        self.state = State(
            (Package("box", (1, 1, 1), 10, 1, False, 1),),
            Truck((2, 2, 2), 5),
            {"box": Placement((0, 0, 0), (1, 1, 1))},
        )

    def result(self, metrics):
        return SearchResult(
            algorithm="algorithm",
            objective_name="total_value",
            initial_state=self.state,
            final_state=self.state,
            objective_value=10,
            objective_history=(5, 10),
            duration_seconds=0.25,
            metrics=metrics,
            seed=12,
        )

    def test_hill_climbing_variants_fit_shared_contract(self):
        result = self.result(
            HillClimbingMetrics(
                variant="random_restart",
                iterations=20,
                restarts=3,
                maximum_restarts=5,
                iterations_per_restart=(5, 7, 8),
            )
        )
        self.assertEqual(result.metrics.restarts, 3)
        self.assertEqual(result.objective_history, (5, 10))

    def test_simulated_annealing_metrics_fit_shared_contract(self):
        result = self.result(
            SimulatedAnnealingMetrics(
                iterations=20,
                exp_delta_e_over_temperature=(1.4, 0.25),
                stuck_count=2,
            )
        )
        self.assertEqual(result.metrics.exp_delta_e_over_temperature, (1.4, 0.25))

    def test_genetic_algorithm_metrics_fit_shared_contract(self):
        result = self.result(
            GeneticAlgorithmMetrics(
                generations=20,
                population_size=30,
                best_objective_history=(5, 10),
                average_objective_history=(3.5, 7.25),
            )
        )
        self.assertEqual(result.metrics.population_size, 30)
        self.assertEqual(result.metrics.average_objective_history, (3.5, 7.25))

    def test_negative_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchResult(
                algorithm="algorithm",
                objective_name="total_value",
                initial_state=self.state,
                final_state=self.state,
                objective_value=10,
                objective_history=(),
                duration_seconds=-1,
                metrics=HillClimbingMetrics("steepest", 0),
            )


if __name__ == "__main__":
    unittest.main()
