import unittest
from random import Random

from src.model import Package, Placement, State, Truck
from src.neighbors import _rotate_candidate, random_initial_state, random_neighbor
from src.validation import validate_state


class NeighborGenerationTests(unittest.TestCase):
    def test_initial_state_is_valid_and_seeded(self):
        packages = (
            Package("a", (2, 2, 1), 10, 1, False, 1),
            Package("b", (2, 2, 1), 20, 1, False, 1),
        )
        truck = Truck((4, 4, 3), 10)
        first = random_initial_state(packages, truck, Random(7))
        second = random_initial_state(packages, truck, Random(7))
        self.assertEqual(first, second)
        self.assertEqual(validate_state(first), [])

    def test_neighbor_is_valid_and_changes_state_when_possible(self):
        packages = (Package("a", (1, 1, 1), 10, 1, False, 1),)
        truck = Truck((4, 4, 4), 10)
        initial = random_initial_state(packages, truck, Random(4))
        neighbor = random_neighbor(initial, Random(9))
        self.assertEqual(validate_state(neighbor), [])
        self.assertNotEqual(neighbor, initial)

    def test_empty_package_set_is_a_valid_state(self):
        initial = random_initial_state((), Truck((4, 4, 4), 10), Random(1))
        self.assertEqual(validate_state(initial), [])
        self.assertIs(random_neighbor(initial, Random(1)), initial)

    def test_rotation_neighbor_swaps_two_axes_only(self):
        package = Package("box", (2, 3, 1), 10, 1, False, 1)
        initial = State(
            (package,),
            Truck((5, 5, 5), 10),
            {"box": Placement((0, 0, 0), (2, 3, 1))},
        )
        rotated = _rotate_candidate(initial, Random(2))
        old_orientation = initial.placements["box"].orientation
        new_orientation = rotated.placements["box"].orientation
        self.assertEqual(sum(a != b for a, b in zip(old_orientation, new_orientation)), 2)
        self.assertEqual(validate_state(rotated), [])


if __name__ == "__main__":
    unittest.main()
