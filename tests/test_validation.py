import unittest

from src.model import Package, Placement, State, Truck
from src.validation import validate_state


def package(package_id, *, size=(2, 2, 1), weight=1, fragile=False):
    return Package(package_id, size, 10, weight, fragile, 1)


def state(packages, placements, *, truck_size=(6, 6, 6), capacity=10):
    return State(tuple(packages), Truck(truck_size, capacity), placements)


class StateValidationTests(unittest.TestCase):
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
