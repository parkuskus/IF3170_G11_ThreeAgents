import unittest
from random import Random

from src.contracts import resolve_objective
from src.model import Package, Truck
from src.neighbors import random_initial_state, random_neighbor
from src.results import HillClimbingMetrics, SearchResult
from src.validation import validate_state


class FoundationIntegrationTests(unittest.TestCase):
    def test_state_generation_neighbor_objective_and_result_contract(self):
        packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (1, 1, 1), 20, 1, False, 1),
        )
        initial = random_initial_state(packages, Truck((3, 3, 3), 5), Random(21))
        neighbor = random_neighbor(initial, Random(22))
        self.assertEqual(validate_state(initial), [])
        self.assertEqual(validate_state(neighbor), [])

        objective = resolve_objective(initial)
        result = SearchResult(
            algorithm="foundation-smoke-check",
            objective_name="total_value",
            initial_state=initial,
            final_state=neighbor,
            objective_value=objective(neighbor),
            objective_history=(objective(initial), objective(neighbor)),
            duration_seconds=0.0,
            metrics=HillClimbingMetrics(variant="steepest", iterations=1),
            seed=22,
        )
        self.assertEqual(result.objective_value, objective(result.final_state))


if __name__ == "__main__":
    unittest.main()
