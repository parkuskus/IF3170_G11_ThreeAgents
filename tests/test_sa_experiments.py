import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.run_sa import main
from src.experiments.simulated_annealing import compare_objectives, run_sa_experiments
from src.model import Package, Truck
from src.neighbors import random_initial_state
from src.validation import validate_state

FIXTURE = Path(__file__).parent / "fixtures" / "ga_hard_problem.json"


class SAExperimentTests(unittest.TestCase):
    def setUp(self):
        self.packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (2, 1, 1), 20, 2, True, 0),
            Package("c", (1, 1, 2), 15, 1, False, 3),
        )
        self.truck = Truck((2, 2, 2), 3)

    def test_runner_saves_three_runs_with_plots_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            records = run_sa_experiments(
                self.packages,
                self.truck,
                seeds=(1, 2, 3),
                max_iterations=20,
                initial_temperature=10.0,
                cooling_rate=0.9,
                stuck_patience=5,
                output_dir=output,
            )

            self.assertEqual([record["run_index"] for record in records], [1, 2, 3])
            for record in records:
                for artifact in record["artifacts"].values():
                    self.assertTrue(Path(artifact).exists(), artifact)
                metrics = record["result"]["metrics"]
                self.assertEqual(len(metrics["exp_delta_e_over_temperature"]), 20)
                self.assertIn("stuck_count", metrics)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["run_count"], 3)
            self.assertTrue((output / "summary-runs.png").exists())

    def test_objective_comparison_scores_every_objective_on_total_value(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            comparisons = compare_objectives(
                self.packages,
                self.truck,
                objective_names=("total_value", "compact_value"),
                output_dir=temporary_dir,
                seeds=(1, 2, 3),
                max_iterations=10,
                initial_temperature=5.0,
                cooling_rate=0.9,
            )

            self.assertEqual(
                [comparison["objective_name"] for comparison in comparisons],
                ["total_value", "compact_value"],
            )
            self.assertTrue(Path(temporary_dir, "objective-comparison.png").exists())
            for comparison in comparisons:
                self.assertIn("total_package_value_mean", comparison)

    def test_runner_rejects_unknown_objective(self):
        with self.assertRaises(ValueError):
            run_sa_experiments(
                self.packages,
                self.truck,
                seeds=(1, 2, 3),
                max_iterations=1,
                initial_temperature=1.0,
                cooling_rate=0.9,
                objective_name="nope",
            )

    def test_cli_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            exit_code = main(
                [
                    "--input", str(FIXTURE),
                    "--output", temporary_dir,
                    "--seeds", "1", "2", "3",
                    "--max-iterations", "5",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(temporary_dir, "total_value", "summary.json").exists())


class SAGAIntegrationTests(unittest.TestCase):
    """SA and GA must give interchangeable results on the same starting state."""

    def test_same_case_yields_valid_consistent_results(self):
        packages = (
            Package("a", (1, 1, 1), 10, 1, False, 1),
            Package("b", (2, 1, 1), 20, 2, True, 0),
            Package("c", (1, 1, 2), 15, 1, False, 3),
            Package("d", (1, 2, 1), 25, 2, False, 2),
        )
        initial = random_initial_state(packages, Truck((2, 2, 2), 4), Random(5))
        results = (
            simulated_annealing(
                initial, seed=5, max_iterations=50, initial_temperature=10.0, cooling_rate=0.95
            ),
            genetic_algorithm(initial, seed=5, population_size=6, generations=5),
        )

        for result in results:
            with self.subTest(algorithm=result.algorithm):
                self.assertEqual(result.initial_state, initial)
                self.assertEqual(validate_state(result.final_state), [])
                self.assertEqual(result.objective_name, "total_value")
                self.assertEqual(result.objective_value, result.final_state.objective())
                self.assertGreaterEqual(result.objective_value, initial.objective())
                # GA logs each generation's maximum (not monotonic); SA logs
                # best-so-far. Both peak at the reported objective.
                self.assertEqual(max(result.objective_history), result.objective_value)
                self.assertGreaterEqual(result.duration_seconds, 0)
                self.assertEqual(result.seed, 5)


if __name__ == "__main__":
    unittest.main()
