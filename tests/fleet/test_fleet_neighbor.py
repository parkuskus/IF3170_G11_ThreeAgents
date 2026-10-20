import unittest
from pathlib import Path
from random import Random

from src.algorithms.genetic_algorithm.fleet import mutate_fleet_individual, random_initial_fleet_state
from src.experiments.common.data import load_fleet_problem
from src.model import FleetState, Package, Placement, Truck
from src.search_problems import search_problem_for_state
from src.validation import validate_fleet_state


class SwapRandom(Random):
    def __init__(self, package_ids):
        super().__init__(11)
        self.package_ids = package_ids

    def randrange(self, *args):
        return 0

    def sample(self, population, k):
        return [next(p for p in population if p.id == name) for name in self.package_ids]


class FleetNeighborTests(unittest.TestCase):
    def test_existing_tight_capacity_has_an_improving_one_step_swap(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "fleet" / "tight-capacity.json"
        packages, trucks = load_fleet_problem(fixture)
        state = random_initial_fleet_state(packages, trucks, Random(11 ^ 0x5EED))
        result = search_problem_for_state(state).neighbor(state, SwapRandom(("P03", "P09")))
        self.assertEqual(state.objective(), 300)
        self.assertEqual(result.objective(), 330)
        self.assertEqual(validate_fleet_state(result), [])

    def test_full_truck_can_replace_a_loaded_package_atomically(self):
        packages = (
            Package("old", (1, 1, 1), 5, 1, False, 0),
            Package("better", (1, 1, 1), 20, 2, False, 0),
            Package("other", (1, 1, 1), 7, 1, False, 0),
        )
        state = FleetState(
            packages,
            (Truck((1, 1, 1), 2), Truck((1, 1, 1), 1)),
            (
                {"old": Placement((0, 0, 0), (1, 1, 1)), "better": None, "other": None},
                {"old": None, "better": None, "other": Placement((0, 0, 0), (1, 1, 1))},
            ),
        )
        result = search_problem_for_state(state).neighbor(state, SwapRandom(("old", "better")))
        self.assertEqual(result.objective(), 27)
        self.assertIsNone(result.placements[0]["old"])
        self.assertIsNotNone(result.placements[0]["better"])
        self.assertEqual(result.placements[1]["other"], state.placements[1]["other"])
        self.assertEqual(validate_fleet_state(result), [])
        self.assertIsNone(mutate_fleet_individual(state, SwapRandom(("old", "better"))).placements[0]["better"])

    def test_two_loaded_packages_swap_trucks_without_duplication(self):
        packages = (
            Package("wide", (2, 1, 1), 10, 2, False, 0),
            Package("small", (1, 1, 1), 20, 1, False, 0),
        )
        state = FleetState(
            packages,
            (Truck((2, 1, 1), 2), Truck((2, 1, 1), 2)),
            (
                {"wide": Placement((0, 0, 0), (2, 1, 1)), "small": None},
                {"wide": None, "small": Placement((0, 0, 0), (1, 1, 1))},
            ),
        )
        result = search_problem_for_state(state).neighbor(state, SwapRandom(("wide", "small")))
        self.assertEqual(validate_fleet_state(result), [])
        self.assertIsNotNone(result.placements[1]["wide"])
        self.assertIsNotNone(result.placements[0]["small"])
        self.assertIsNone(result.placements[0]["wide"])
        self.assertIsNone(result.placements[1]["small"])

    def test_impossible_replacement_never_returns_invalid_fleet(self):
        packages = (
            Package("old", (1, 1, 1), 5, 1, False, 0),
            Package("huge", (3, 3, 3), 20, 2, False, 0),
        )
        state = FleetState(
            packages,
            (Truck((1, 1, 1), 2),),
            ({"old": Placement((0, 0, 0), (1, 1, 1)), "huge": None},),
        )
        result = search_problem_for_state(state).neighbor(state, SwapRandom(("old", "huge")))
        self.assertEqual(validate_fleet_state(result), [])
        self.assertIsNone(result.placements[0]["huge"])

    def test_swap_does_not_place_fragile_package_under_existing_stack(self):
        packages = (
            Package("support", (1, 1, 1), 5, 1, False, 0),
            Package("fragile", (1, 1, 1), 20, 1, True, 0),
            Package("top", (1, 1, 1), 7, 1, False, 0),
        )
        state = FleetState(
            packages,
            (Truck((1, 1, 2), 2),),
            ({"support": Placement((0, 0, 0), (1, 1, 1)),
              "fragile": None,
              "top": Placement((0, 0, 1), (1, 1, 1))},),
        )
        result = search_problem_for_state(state).neighbor(
            state, SwapRandom(("support", "fragile"))
        )
        self.assertIsNone(result.placements[0]["fragile"])
        self.assertEqual(validate_fleet_state(result), [])

    def test_single_package_and_empty_fleet_packages_remain_valid(self):
        truck = Truck((1, 1, 1), 1)
        for packages, placements in (
            ((), {}),
            ((Package("one", (1, 1, 1), 1, 1, False, 0),), {"one": Placement((0, 0, 0), (1, 1, 1))}),
        ):
            state = FleetState(packages, (truck,), (placements,))
            neighbor = search_problem_for_state(state).neighbor(state, Random(11))
            self.assertEqual(validate_fleet_state(neighbor), [])

    def test_seeded_swap_repeats(self):
        packages = (
            Package("old", (1, 1, 1), 5, 1, False, 0),
            Package("better", (1, 1, 1), 20, 2, False, 0),
        )
        state = FleetState(packages, (Truck((1, 1, 1), 2),),
                           ({"old": Placement((0, 0, 0), (1, 1, 1)), "better": None},))
        problem = search_problem_for_state(state)
        self.assertEqual(problem.neighbor(state, Random(21)), problem.neighbor(state, Random(21)))


if __name__ == "__main__":
    unittest.main()
