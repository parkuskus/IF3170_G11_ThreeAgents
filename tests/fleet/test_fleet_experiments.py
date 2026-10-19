import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.run_fleet_ga import main


class FleetExperimentTests(unittest.TestCase):
    def test_cli_saves_heterogeneous_fleet_result_and_visualizations(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            problem = root / "fleet.json"
            output = root / "output"
            problem.write_text(json.dumps({
                "packages": [
                    {"id": "a", "dimensions": [1, 1, 1], "value": 10, "weight": 2, "fragile": False, "eta": 0},
                    {"id": "b", "dimensions": [1, 1, 1], "value": 20, "weight": 2, "fragile": False, "eta": 0},
                ],
                "trucks": [
                    {"dimensions": [1, 1, 1], "max_capacity": 2},
                    {"dimensions": [2, 1, 1], "max_capacity": 2},
                ],
            }), encoding="utf-8")

            result = main([
                "--input", str(problem), "--output", str(output),
                "--population-size", "3", "--generations", "1", "--seed", "11",
            ])

            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(record["result"]["final_state"]["total_package_value"], 30)
            trucks = record["result"]["final_state"]["trucks"]
            self.assertEqual(len(trucks), 2)
            self.assertEqual([truck["dimensions"] for truck in trucks], [[1, 1, 1], [2, 1, 1]])
            self.assertCountEqual(
                [package["id"] for truck in trucks for package in truck["packages"]],
                ["a", "b"],
            )
            self.assertTrue((output / "objective-history.png").is_file())
            self.assertTrue((output / "initial-final-states.png").is_file())

    def test_cli_runs_seed_and_objective_comparison(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            problem = root / "fleet.json"
            problem.write_text(json.dumps({
                "packages": [
                    {"id": "a", "dimensions": [1, 1, 1], "value": 10, "weight": 1, "fragile": False, "eta": 0},
                    {"id": "b", "dimensions": [1, 1, 1], "value": 20, "weight": 1, "fragile": False, "eta": 2},
                ],
                "trucks": [
                    {"dimensions": [1, 1, 1], "max_capacity": 1},
                    {"dimensions": [1, 1, 1], "max_capacity": 1},
                ],
            }), encoding="utf-8")
            output = root / "output"

            self.assertEqual(main([
                "--input", str(problem), "--output", str(output),
                "--population-size", "2", "--generations", "1",
                "--seeds", "7", "8", "9",
                "--objectives", "total_value", "eta_weighted_value",
            ]), 0)

            records = json.loads((output / "results.json").read_text(encoding="utf-8"))
            comparison = json.loads(
                (output / "objective-comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(records), 6)
            self.assertEqual({record["seed"] for record in records}, {7, 8, 9})
            self.assertEqual({item["objective_name"] for item in comparison}, {
                "total_value", "eta_weighted_value"
            })
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
