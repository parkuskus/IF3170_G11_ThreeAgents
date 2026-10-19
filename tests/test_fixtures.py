"""Check that the fixtures realise the instance categories of the report methodology."""

import unittest
from itertools import combinations
from pathlib import Path

from src.experiments.data import load_problem
from src.model import ORIENTATIONS, Placement, State
from src.validation import validate_state

FIXTURES = Path(__file__).parent / "fixtures"


def _fits_alone(package, truck):
    return package.weight <= truck.max_capacity and any(
        all(size <= limit for size, limit in zip(package.oriented_dimensions(o), truck.dimensions))
        for o in ORIENTATIONS
    )


class FixtureCategoryTests(unittest.TestCase):
    def test_every_valid_fixture_loads(self):
        paths = [
            *FIXTURES.glob("*.json"),
            *(FIXTURES / "boundary").glob("*.json"),
        ]
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            with self.subTest(path=path.name):
                packages, _ = load_problem(path)
                self.assertGreater(len(packages), 0)

    def test_every_invalid_fixture_is_rejected(self):
        paths = sorted((FIXTURES / "invalid").glob("*.json"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                with self.assertRaises(ValueError):
                    load_problem(path)

    def test_base_requires_choosing_a_subset(self):
        packages, truck = load_problem(FIXTURES / "base.json")
        self.assertGreater(sum(p.weight for p in packages), truck.max_capacity)
        self.assertTrue(all(_fits_alone(p, truck) for p in packages))
        self.assertGreater(len({p.eta for p in packages}), 3)
        self.assertTrue(any(p.fragile for p in packages))

    def test_tight_cases_change_exactly_one_factor_of_base(self):
        base_packages, base_truck = load_problem(FIXTURES / "base.json")
        space_packages, space_truck = load_problem(FIXTURES / "tight-space.json")
        cap_packages, cap_truck = load_problem(FIXTURES / "tight-capacity.json")
        self.assertEqual(space_packages, base_packages)
        self.assertEqual(cap_packages, base_packages)
        self.assertEqual(space_truck.max_capacity, base_truck.max_capacity)
        self.assertEqual(cap_truck.dimensions, base_truck.dimensions)
        volume = lambda t: t.dimensions[0] * t.dimensions[1] * t.dimensions[2]
        self.assertLess(volume(space_truck), volume(base_truck))
        self.assertLess(cap_truck.max_capacity, base_truck.max_capacity)

    def test_infeasible_has_no_loadable_package(self):
        packages, truck = load_problem(FIXTURES / "infeasible.json")
        base_packages, base_truck = load_problem(FIXTURES / "base.json")
        self.assertEqual(truck, base_truck)
        self.assertEqual(len(packages), len(base_packages))
        self.assertFalse(any(_fits_alone(p, truck) for p in packages))
        self.assertTrue(all(p.value > 0 for p in packages))

    def test_rotation_only_package_needs_rotation(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "rotation-only.json")
        long_package = next(p for p in packages if p.id == "long")
        self.assertFalse(
            all(s <= l for s, l in zip(long_package.dimensions, truck.dimensions))
        )
        self.assertTrue(_fits_alone(long_package, truck))

    def test_wall_touching_packages_fill_the_truck_exactly(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "wall-touching.json")
        volume = lambda dims: dims[0] * dims[1] * dims[2]
        self.assertEqual(sum(volume(p.dimensions) for p in packages), volume(truck.dimensions))
        placements = {
            p.id: Placement((2 * (i % 2), 2 * (i // 2), 0), p.dimensions)
            for i, p in enumerate(packages)
        }
        self.assertEqual(validate_state(State(packages, truck, placements)), [])

    def test_exact_capacity_optimum_equals_capacity(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "exact-capacity.json")
        feasible = [
            subset
            for size in range(len(packages) + 1)
            for subset in combinations(packages, size)
            if sum(p.weight for p in subset) <= truck.max_capacity
        ]
        best = max(feasible, key=lambda subset: sum(p.value for p in subset))
        self.assertEqual(sum(p.weight for p in best), truck.max_capacity)

    def test_fragile_package_may_be_topped_but_not_stacked_under(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "fragile-support.json")
        by_id = {p.id: p for p in packages}
        floor, top = (0, 0, 0), (0, 0, 1)
        cell = lambda p: (p.dimensions)
        on_solid = State(
            packages,
            truck,
            {
                "solid-a": Placement(floor, cell(by_id["solid-a"])),
                "fragile": Placement(top, cell(by_id["fragile"])),
                "solid-b": None,
            },
        )
        under_fragile = State(
            packages,
            truck,
            {
                "fragile": Placement(floor, cell(by_id["fragile"])),
                "solid-a": Placement(top, cell(by_id["solid-a"])),
                "solid-b": None,
            },
        )
        self.assertEqual(validate_state(on_solid), [])
        self.assertTrue(validate_state(under_fragile))

    def test_zero_value_and_zero_weight_packages_are_present(self):
        packages, _ = load_problem(FIXTURES / "boundary" / "zero-value-weight.json")
        self.assertTrue(any(p.value == 0 and p.weight > 0 for p in packages))
        self.assertTrue(any(p.weight == 0 and p.value > 0 for p in packages))
        self.assertTrue(any(p.value == 0 and p.weight == 0 for p in packages))

    def test_equal_fitness_packages_are_interchangeable(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "equal-fitness.json")
        signatures = {(p.dimensions, p.value, p.weight, p.fragile, p.eta) for p in packages}
        self.assertEqual(len(signatures), 1)
        self.assertGreater(len(packages), truck.dimensions[0] * truck.dimensions[1] * truck.dimensions[2])

    def test_single_package_fixture_has_one_package(self):
        packages, truck = load_problem(FIXTURES / "boundary" / "single-package.json")
        self.assertEqual(len(packages), 1)
        self.assertTrue(_fits_alone(packages[0], truck))


if __name__ == "__main__":
    unittest.main()
