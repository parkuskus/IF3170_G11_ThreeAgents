import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.run_fleet_hc import main as run_hc
from src.experiments.run_fleet_sa import main as run_sa


class FleetLocalSearchCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.input_path = self.root / "fleet.json"
        self.input_path.write_text(
            json.dumps(
                {
                    "packages": [
                        {
                            "id": "a", "dimensions": [1, 1, 1], "value": 10,
                            "weight": 1, "fragile": False, "eta": 0,
                        },
                        {
                            "id": "b", "dimensions": [1, 1, 1], "value": 20,
                            "weight": 1, "fragile": False, "eta": 0,
                        },
                    ],
                    "trucks": [
                        {"dimensions": [1, 1, 1], "max_capacity": 1},
                        {"dimensions": [1, 1, 1], "max_capacity": 1},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_hill_climbing_cli_writes_fleet_result_and_artifacts(self):
        output = self.root / "hc"
        self.assertEqual(
            run_hc(
                ["--input", str(self.input_path), "--output", str(output),
                 "--variant", "steepest", "--max-iterations", "2", "--seed", "7"]
            ),
            0,
        )
        record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["result"]["algorithm"], "hill_climbing_steepest")
        self.assertEqual(record["result"]["final_state"]["truck_count"], 2)
        self.assertTrue(Path(record["artifacts"]["objective_history"]).is_file())
        self.assertTrue(Path(record["artifacts"]["initial_final_states"]).is_file())

    def test_simulated_annealing_cli_writes_fleet_result_and_artifacts(self):
        output = self.root / "sa"
        self.assertEqual(
            run_sa(
                ["--input", str(self.input_path), "--output", str(output),
                 "--max-iterations", "2", "--seed", "7"]
            ),
            0,
        )
        record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["result"]["algorithm"], "simulated_annealing")
        self.assertEqual(record["result"]["final_state"]["truck_count"], 2)
        self.assertTrue(Path(record["artifacts"]["objective_history"]).is_file())
        self.assertTrue(Path(record["artifacts"]["initial_final_states"]).is_file())

    def test_hill_climbing_cli_runs_seed_and_objective_comparison(self):
        output = self.root / "hc-sweep"
        run_hc(
            ["--input", str(self.input_path), "--output", str(output),
             "--variant", "steepest", "--max-iterations", "1",
             "--seeds", "7", "8", "9",
             "--objectives", "total_value", "compact_value"]
        )
        records = json.loads((output / "total_value" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual([record["seed"] for record in records], [7, 8, 9])
        comparison = json.loads((output / "objective-comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(len(comparison), 2)

    def test_simulated_annealing_cli_runs_seed_and_objective_comparison(self):
        output = self.root / "sa-sweep"
        run_sa(
            ["--input", str(self.input_path), "--output", str(output),
             "--max-iterations", "1", "--seeds", "7", "8", "9",
             "--objectives", "total_value", "compact_value"]
        )
        records = json.loads((output / "total_value" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual([record["seed"] for record in records], [7, 8, 9])
        comparison = json.loads((output / "objective-comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(len(comparison), 2)


if __name__ == "__main__":
    unittest.main()
