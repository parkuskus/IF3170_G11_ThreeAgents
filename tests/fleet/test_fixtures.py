"""Ensure fleet fixtures preserve the existing synthetic test cases."""

import json
import unittest
from pathlib import Path

from src.experiments.common.benchmark import discover_instances
from src.experiments.common.data import load_fleet_problem, load_problem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FLEET_FIXTURES = FIXTURES / "fleet"


class FleetFixtureTests(unittest.TestCase):
    def test_fleet_fixtures_mirror_all_valid_cases_with_two_identical_trucks(self):
        instances = discover_instances(FIXTURES)
        expected_names = {name for _, name, _ in instances}
        found_names = {
            path.stem
            for path in FLEET_FIXTURES.glob("*.json")
        } | {
            path.stem
            for path in (FLEET_FIXTURES / "boundary").glob("*.json")
        }
        self.assertEqual(found_names, expected_names)

        for _, name, source_path in instances:
            relative = source_path.relative_to(FIXTURES)
            fleet_path = FLEET_FIXTURES / relative
            with self.subTest(fixture=name):
                source_packages, source_truck = load_problem(source_path)
                fleet_packages, fleet_trucks = load_fleet_problem(fleet_path)
                self.assertEqual(fleet_packages, source_packages)
                self.assertEqual(fleet_trucks, (source_truck, source_truck))
                data = json.loads(fleet_path.read_text(encoding="utf-8"))
                self.assertEqual(len(data["trucks"]), 2)


if __name__ == "__main__":
    unittest.main()
