import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from random import Random
from unittest.mock import patch

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.experiments.genetic_algorithm import run_ga_experiments
from src.experiments.edge_cases import run_edge_case_comparison
from src.model import Package, Truck
from src.neighbors import random_initial_state
from src.visualization.genetic_algorithm import (
    plot_ga_history,
    plot_state_comparison,
)


class GAExperimentTests(unittest.TestCase):
    def setUp(self):
        self.packages = (Package("box", (1, 1, 1), 10, 1, False, 1),)
        self.truck = Truck((2, 2, 2), 5)

    def test_runner_executes_both_three_by_three_sweeps(self):
        with tempfile.TemporaryDirectory() as temporary_dir, patch(
            "src.experiments.genetic_algorithm.plot_ga_history"
        ) as history_plot, patch(
            "src.experiments.genetic_algorithm.plot_state_comparison"
        ) as state_plot:
            records = run_ga_experiments(
                self.packages,
                self.truck,
                generation_values=(0, 1, 2),
                population_values=(1, 2, 3),
                fixed_population_size=1,
                fixed_generations=1,
                seeds=(11, 12, 13),
                output_dir=temporary_dir,
            )

            self.assertEqual(len(records), 18)
            self.assertEqual(history_plot.call_count, 18)
            self.assertEqual(state_plot.call_count, 18)
            counts = Counter(
                (record["experiment"], record["population_size"], record["generations"])
                for record in records
            )
            self.assertEqual(len(counts), 6)
            self.assertTrue(all(count == 3 for count in counts.values()))
            self.assertEqual(len(json.loads((Path(temporary_dir) / "results.json").read_text())), 18)
            summaries = json.loads((Path(temporary_dir) / "summary.json").read_text())
            self.assertEqual(len(summaries), 6)
            self.assertTrue(all(summary["run_count"] == 3 for summary in summaries))
            self.assertTrue(
                (Path(temporary_dir) / "summary-generation-sweep.png").is_file()
            )
            self.assertTrue(
                (Path(temporary_dir) / "summary-population-sweep.png").is_file()
            )

    def test_runner_rejects_incomplete_parameter_sweep(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                run_ga_experiments(
                    self.packages,
                    self.truck,
                    generation_values=(1, 2),
                    population_values=(1, 2, 3),
                    fixed_population_size=2,
                    fixed_generations=2,
                    seeds=(1, 2, 3),
                    output_dir=temporary_dir,
                )

    def test_visualizers_save_history_and_state_images(self):
        initial = random_initial_state(self.packages, self.truck, Random(5))
        result = genetic_algorithm(
            initial, seed=7, population_size=2, generations=1
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            history = Path(temporary_dir) / "history.png"
            states = Path(temporary_dir) / "states.png"
            plot_ga_history(result, history)
            plot_state_comparison(result.initial_state, result.final_state, states)
            self.assertGreater(history.stat().st_size, 0)
            self.assertGreater(states.stat().st_size, 0)

    def test_edge_case_runner_creates_comparison_curve(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summaries = run_edge_case_comparison(
                self.packages,
                self.truck,
                seeds=(21, 22, 23),
                population_size=2,
                generations=1,
                output_dir=temporary_dir,
            )
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(summary["run_count"] == 3 for summary in summaries))
            self.assertTrue(
                (Path(temporary_dir) / "objective-vs-generations.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
