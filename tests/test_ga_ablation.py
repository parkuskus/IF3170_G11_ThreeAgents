import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.ablation import run_ga_ablation
from src.model import Package, Truck


class GAAblationTests(unittest.TestCase):
    def test_runs_baseline_and_three_single_component_ablations(self):
        packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (1, 1, 1), 20, 1, False, 1),
        )
        truck = Truck((3, 3, 3), 5)
        with tempfile.TemporaryDirectory() as temp_dir:
            records = run_ga_ablation(
                packages,
                truck,
                seeds=(41, 42),
                population_size=4,
                generations=3,
                output_dir=temp_dir,
            )
            self.assertEqual(len(records), 8)
            by_variant = {name: [r for r in records if r["variant"] == name] for name in {
                "baseline", "no-mutation", "no-crossover", "uniform-selection"
            }}
            self.assertTrue(all(len(runs) == 2 for runs in by_variant.values()))
            self.assertEqual(by_variant["baseline"][0]["parameters"]["mutation_rate"], 0.05)
            self.assertEqual(by_variant["no-mutation"][0]["parameters"]["mutation_rate"], 0)
            self.assertFalse(by_variant["no-crossover"][0]["parameters"]["crossover_enabled"])
            self.assertEqual(by_variant["uniform-selection"][0]["parameters"]["selection_mode"], "uniform")
            self.assertEqual(len(json.loads((Path(temp_dir) / "runs.json").read_text())), 8)
            self.assertTrue((Path(temp_dir) / "summary.json").is_file())
            self.assertTrue((Path(temp_dir) / "objective-comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
