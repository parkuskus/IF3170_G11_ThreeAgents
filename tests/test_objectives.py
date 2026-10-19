import unittest

from src.model import Package, Placement, State, Truck
from src.objectives import (
    OBJECTIVES,
    compact_value,
    compactness,
    eta_weighted_value,
    total_value,
)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.packages = (
            Package("urgent", (1, 1, 1), 30, 1, False, 0),
            Package("later", (1, 1, 1), 30, 1, False, 2),
        )
        self.truck = Truck((2, 2, 4), 10)

    def state(self, **placements):
        all_placements = {package.id: None for package in self.packages}
        all_placements.update(placements)
        return State(self.packages, self.truck, all_placements)

    def test_total_value_matches_required_objective(self):
        state = self.state(urgent=Placement((0, 0, 0), (1, 1, 1)))
        self.assertEqual(total_value(state), state.objective())

    def test_eta_weighted_value_prefers_urgent_packages(self):
        urgent = self.state(urgent=Placement((0, 0, 0), (1, 1, 1)))
        later = self.state(later=Placement((0, 0, 0), (1, 1, 1)))

        self.assertEqual(eta_weighted_value(urgent), 30)
        self.assertEqual(eta_weighted_value(later), 10)

    def test_compactness_rewards_lower_stacking(self):
        floor = self.state(
            urgent=Placement((0, 0, 0), (1, 1, 1)),
            later=Placement((1, 0, 0), (1, 1, 1)),
        )
        stacked = self.state(
            urgent=Placement((0, 0, 0), (1, 1, 1)),
            later=Placement((0, 0, 1), (1, 1, 1)),
        )

        self.assertAlmostEqual(compactness(floor), 0.75)
        self.assertAlmostEqual(compactness(stacked), 0.625)
        self.assertEqual(compactness(self.state()), 0.0)
        self.assertGreater(compact_value(floor), compact_value(stacked))

    def test_compact_value_never_outranks_a_higher_total_value(self):
        one_low = self.state(urgent=Placement((0, 0, 0), (1, 1, 1)))
        two_high = self.state(
            urgent=Placement((0, 0, 0), (1, 1, 1)),
            later=Placement((0, 0, 1), (1, 1, 1)),
        )

        self.assertGreater(compact_value(two_high), compact_value(one_low))
        self.assertLess(compact_value(one_low) - total_value(one_low), 1)

    def test_registry_exposes_required_and_alternative_objectives(self):
        self.assertEqual(
            set(OBJECTIVES), {"total_value", "eta_weighted_value", "compact_value"}
        )


if __name__ == "__main__":
    unittest.main()
