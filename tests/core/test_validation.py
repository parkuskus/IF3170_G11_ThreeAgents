import unittest

from src.model import FleetState, Package, Placement, State, Truck
from src.validation import validate_fleet_state, validate_state


def package(package_id, *, size=(2, 2, 1), weight=1, fragile=False):
    return Package(package_id, size, 10, weight, fragile, 1)


def state(packages, placements, *, truck_size=(6, 6, 6), capacity=10):
    return State(tuple(packages), Truck(truck_size, capacity), placements)


class StateValidationTests(unittest.TestCase):
    def test_fleet_rejects_package_placed_in_multiple_trucks(self):
        item = package("p", size=(1, 1, 1))
        fleet = FleetState(
            (item,),
            (Truck((2, 2, 2), 2), Truck((3, 3, 3), 2)),
            (
                {"p": Placement((0, 0, 0), (1, 1, 1))},
                {"p": Placement((0, 0, 0), (1, 1, 1))},
            ),
        )
        self.assertTrue(any("more than one truck" in error for error in validate_fleet_state(fleet)))

    def test_fleet_validates_each_heterogeneous_truck_independently(self):
        item = package("p", size=(2, 1, 1), weight=2)
        fleet = FleetState(
            (item,),
            (Truck((1, 1, 1), 2), Truck((3, 2, 2), 2)),
            ({"p": None}, {"p": Placement((0, 0, 0), (2, 1, 1))}),
        )
        self.assertEqual(validate_fleet_state(fleet), [])

    def test_fleet_reports_invalid_geometry_on_affected_truck(self):
        item = package("p", size=(2, 1, 1), weight=2)
        fleet = FleetState(
            (item,),
            (Truck((3, 2, 2), 2), Truck((1, 1, 1), 2)),
            ({"p": None}, {"p": Placement((0, 0, 0), (2, 1, 1))}),
        )
        errors = validate_fleet_state(fleet)
        self.assertTrue(any("truck 1" in error and "outside" in error for error in errors))

    def test_fleet_reports_over_capacity_on_affected_truck(self):
        item = package("p", size=(1, 1, 1), weight=3)
        fleet = FleetState(
            (item,),
            (Truck((1, 1, 1), 5), Truck((1, 1, 1), 2)),
            ({"p": None}, {"p": Placement((0, 0, 0), (1, 1, 1))}),
        )
        self.assertIn("truck 1: truck capacity is exceeded", validate_fleet_state(fleet))

    def test_valid_grounded_state(self):
        packages = (package("base"), package("top"))
        candidate = state(
            packages,
            {
                "base": Placement((0, 0, 0), (2, 2, 1)),
                "top": Placement((0, 0, 1), (2, 2, 1)),
            },
        )
        self.assertEqual(validate_state(candidate), [])

    def test_rejects_package_outside_truck(self):
        candidate = state(
            (package("box"),),
            {"box": Placement((5, 0, 0), (2, 2, 1))},
        )
        self.assertTrue(any("outside the truck" in error for error in validate_state(candidate)))

    def test_rejects_overlapping_packages(self):
        packages = (package("a"), package("b"))
        candidate = state(
            packages,
            {"a": Placement((0, 0, 0), (2, 2, 1)), "b": Placement((1, 1, 0), (2, 2, 1))},
        )
        self.assertTrue(any("overlap" in error for error in validate_state(candidate)))

    def test_rejects_over_capacity(self):
        packages = (package("a", weight=6), package("b", weight=5))
        candidate = state(
            packages,
            {"a": Placement((0, 0, 0), (2, 2, 1)), "b": Placement((2, 0, 0), (2, 2, 1))},
            capacity=10,
        )
        self.assertIn("truck capacity is exceeded", validate_state(candidate))

    def test_rejects_floating_package(self):
        candidate = state(
            (package("top"),),
            {"top": Placement((0, 0, 2), (2, 2, 1))},
        )
        self.assertTrue(any("unsupported" in error for error in validate_state(candidate)))

    def test_fragile_package_cannot_support_another(self):
        packages = (package("fragile", fragile=True), package("top"))
        candidate = state(
            packages,
            {
                "fragile": Placement((0, 0, 0), (2, 2, 1)),
                "top": Placement((0, 0, 1), (2, 2, 1)),
            },
        )
        self.assertTrue(any("'top' is unsupported" in error for error in validate_state(candidate)))


if __name__ == "__main__":
    unittest.main()
