import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.ga.experiments import compare_ga_objectives, run_ga_experiments
from src.experiments.hc.experiments import compare_hc_objectives, run_hc_experiments
from src.experiments.run_ga import main as run_ga_main
from src.experiments.run_hill_climbing import main as run_hc_main
from src.model import Package, Truck

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hard.json"
OBJECTIVES = ("total_value", "eta_weighted_value", "compact_value")


class AlternativeObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (2, 1, 1), 20, 2, True, 0),
            Package("c", (1, 1, 2), 15, 1, False, 3),
        )
        self.truck = Truck((2, 2, 2), 3)

    def test_hc_comparison_covers_every_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparisons = compare_hc_objectives(
                self.packages, self.truck, objective_names=OBJECTIVES, output_dir=tmp,
                seeds=(1, 2, 3), variants=("steepest",), max_iterations=5,
                neighbors_per_iteration=3,
            )
            self.assertEqual([c["objective_name"] for c in comparisons], list(OBJECTIVES))
            self.assertTrue(Path(tmp, "objective-comparison.png").exists())
            records = json.loads(
                Path(tmp, "eta_weighted_value", "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(records[0]["result"]["objective_name"], "eta_weighted_value")

    def test_ga_comparison_covers_every_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparisons = compare_ga_objectives(
                self.packages, self.truck, objective_names=OBJECTIVES, output_dir=tmp,
                generation_values=(0, 1, 2), population_values=(2, 3, 4),
                fixed_population_size=3, fixed_generations=1, seeds=(1, 2, 3),
            )
            self.assertEqual([c["objective_name"] for c in comparisons], list(OBJECTIVES))
            self.assertTrue(Path(tmp, "objective-comparison.json").exists())

    def test_runners_reject_unknown_objective(self):
        with self.assertRaises(ValueError):
            run_hc_experiments(self.packages, self.truck, seeds=(1, 2, 3), objective_name="nope")
        with self.assertRaises(ValueError):
            run_ga_experiments(
                self.packages, self.truck, generation_values=(0, 1, 2),
                population_values=(1, 2, 3), fixed_population_size=2, fixed_generations=1,
                seeds=(1, 2, 3), objective_name="nope",
            )

    def test_clis_accept_multiple_objectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_hc_main([
                "--input", str(FIXTURE), "--output", tmp, "--seeds", "1", "2", "3",
                "--variants", "steepest", "--max-iterations", "3",
                "--neighbors-per-iteration", "2",
                "--objectives", "total_value", "compact_value",
            ]), 0)
            self.assertTrue(Path(tmp, "compact_value", "summary.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_ga_main([
                "--input", str(FIXTURE), "--output", tmp, "--seeds", "1", "2", "3",
                "--generation-values", "0", "1", "2", "--population-values", "2", "3", "4",
                "--fixed-population-size", "3", "--fixed-generations", "1",
                "--objectives", "total_value", "eta_weighted_value",
            ]), 0)
            self.assertTrue(Path(tmp, "eta_weighted_value", "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
