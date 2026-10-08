import unittest
from random import Random

from src.model import Package, Truck
from src.neighbors import random_initial_state, random_neighbor
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


if __name__ == "__main__":
    unittest.main()
