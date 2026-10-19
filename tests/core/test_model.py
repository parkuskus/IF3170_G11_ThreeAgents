import unittest

from src.model import FleetState, ORIENTATIONS, Package, Placement, State, Truck


class DomainModelTests(unittest.TestCase):
    def setUp(self):
        self.package = Package("box", (2, 3, 4), 60, 5, False, 3)

    def test_all_six_orientations_are_available(self):
        self.assertEqual(len(ORIENTATIONS), 6)
        self.assertEqual(
            {self.package.oriented_dimensions(orientation) for orientation in ORIENTATIONS},
            {(2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2)},
        )

    def test_state_objective_counts_only_placed_packages(self):
        state = State(
            packages=(self.package, Package("outside", (1, 1, 1), 10, 1, False, 1)),
            truck=Truck((10, 20, 5), 500),
            placements={"box": Placement((0, 0, 0), (2, 3, 4)), "outside": None},
        )
        self.assertEqual(state.objective(), 60)

    def test_fleet_objective_sums_packages_across_trucks(self):
        outside = Package("outside", (1, 1, 1), 10, 1, False, 1)
        fleet = FleetState(
            (self.package, outside),
            (Truck((10, 20, 5), 500), Truck((5, 5, 5), 10)),
            (
                {"box": Placement((0, 0, 0), (2, 3, 4)), "outside": None},
                {"box": None, "outside": Placement((0, 0, 0), (1, 1, 1))},
            ),
        )
        self.assertEqual(fleet.objective(), 70)

    def test_fleet_requires_a_placement_map_per_truck(self):
        with self.assertRaises(ValueError):
            FleetState((self.package,), (Truck((3, 3, 3), 5),), ())

    def test_fleet_requires_at_least_one_truck(self):
        with self.assertRaisesRegex(ValueError, "at least one truck"):
            FleetState((self.package,), (), ())

    def test_state_requires_placement_for_every_package(self):
        with self.assertRaises(ValueError):
            State((self.package,), Truck((10, 20, 5), 500), {})

    def test_state_rejects_duplicate_package_ids(self):
        duplicate = Package("box", (1, 1, 1), 1, 1, False, 0)
        with self.assertRaises(ValueError):
            State((self.package, duplicate), Truck((10, 20, 5), 500), {"box": None})

    def test_state_rejects_oriented_dimensions_that_do_not_match(self):
        with self.assertRaises(ValueError):
            State(
                (self.package,),
                Truck((10, 20, 5), 500),
                {"box": Placement((0, 0, 0), (2, 3, 4), (1, 0, 2))},
            )

    def test_invalid_package_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            Package("bad", (2, 0, 4), 1, 1, False, 0)


if __name__ == "__main__":
    unittest.main()
