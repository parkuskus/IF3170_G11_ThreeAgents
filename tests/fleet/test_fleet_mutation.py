import unittest
from pathlib import Path
from random import Random

from src.algorithms.genetic_algorithm.fleet import (
    mutate_fleet_individual,
    random_initial_fleet_state,
)
from src.experiments.common.data import load_fleet_problem
from src.validation import validate_fleet_state

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FleetMutationValidityTest(unittest.TestCase):
    def test_mutation_never_returns_an_invalid_fleet(self):
        # Removing a package from under a stack used to leave the stack floating.
        packages, trucks = load_fleet_problem(FIXTURES / "demo.json", truck_count=3)
        for seed in range(60):
            rng = Random(seed)
            state = random_initial_fleet_state(packages, trucks, rng)
            for _ in range(30):
                state = mutate_fleet_individual(state, rng)
                self.assertEqual(validate_fleet_state(state), [], f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
