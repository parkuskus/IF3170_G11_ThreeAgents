import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.experiments.common.data import load_fleet_problem, load_problem
from src.experiments.run_ga import main


class GADataTests(unittest.TestCase):
    def test_loads_heterogeneous_fleet_trucks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "fleet.json"
            path.write_text(json.dumps({
                "packages": [{"id": "p", "dimensions": [1, 1, 1], "value": 1, "weight": 1, "fragile": False, "eta": 0}],
                "trucks": [
                    {"dimensions": [3, 2, 1], "max_capacity": 5},
                    {"dimensions": [1, 1, 1], "max_capacity": 1},
                ],
            }), encoding="utf-8")
            packages, trucks = load_fleet_problem(path)
        self.assertEqual(len(packages), 1)
        self.assertEqual([truck.dimensions for truck in trucks], [(3, 2, 1), (1, 1, 1)])

    def test_legacy_truck_can_be_replicated_and_count_is_validated(self):
        problem = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.json"
        _, trucks = load_fleet_problem(problem, truck_count=2)
        self.assertEqual(len(trucks), 2)
        self.assertEqual(trucks[0], trucks[1])
        for invalid_count in (0, -1, True, 1.5):
            with self.subTest(count=invalid_count), self.assertRaises(ValueError):
                load_fleet_problem(problem, truck_count=invalid_count)

    def test_truck_count_cannot_override_explicit_fleet(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "fleet.json"
            path.write_text(json.dumps({
                "packages": [{"id": "p", "dimensions": [1, 1, 1], "value": 1, "weight": 1, "fragile": False, "eta": 0}],
                "trucks": [{"dimensions": [1, 1, 1], "max_capacity": 1}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fleet_problem(path, truck_count=2)

    def test_loads_demo_problem(self):
        path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.json"
        packages, truck = load_problem(path)
        self.assertEqual(len(packages), 5)
        self.assertEqual(truck.max_capacity, 12)

    def test_rejects_invalid_problem_shape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "problem.json"
            path.write_text(json.dumps({"packages": [], "truck": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_problem(path)

    def test_cli_loads_problem_and_dispatches_runner(self):
        problem = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.json"
        with patch("src.experiments.ga.run.run_ga_experiments", return_value=()) as runner:
            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--input", str(problem),
                        "--generation-values", "4", "6", "8",
                        "--population-values", "4", "6", "8",
                        "--fixed-population-size", "6",
                        "--fixed-generations", "6",
                        "--seeds", "41", "42", "43",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(runner.call_count, 1)


if __name__ == "__main__":
    unittest.main()
