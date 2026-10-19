import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from src.experiments.common.benchmark import discover_instances
from src.experiments.common.data import load_problem
from src.experiments.ga.results_analysis import (
    _configurations,
    _run_directory,
    summarize_ga_analysis,
)
from src.model import State
from src.visualization.genetic_algorithm import _draw_state


class GAResultsAnalysisTests(unittest.TestCase):
    def test_each_sweep_configuration_gets_its_own_run_directory(self):
        configurations = _configurations(discover_instances(Path("tests/fixtures")))
        directories = {
            _run_directory(Path("output"), experiment, instance, factor, value, 11)
            for experiment, _, instance, _, factor, value, _ in configurations
        }

        self.assertEqual(len(directories), 20)

    def test_state_visualization_summarizes_long_outside_package_lists(self):
        packages, truck = load_problem(Path("tests/fixtures/infeasible.json"))
        state = State(packages, truck, {package.id: None for package in packages})
        fig = plt.figure()
        try:
            ax = fig.add_subplot(111, projection="3d")
            _draw_state(ax, state, "Empty")
            self.assertIn("10 packages", ax.get_title())
        finally:
            plt.close(fig)

    def test_protocol_covers_all_fixture_baselines_and_three_base_sweeps(self):
        configurations = list(_configurations(discover_instances(Path("tests/fixtures"))))
        baselines = [config for config in configurations if config[0] == "baseline"]
        sweeps = [config for config in configurations if config[0] != "baseline"]

        self.assertEqual(len(baselines), 11)
        self.assertEqual(len(sweeps), 9)
        self.assertTrue(all(config[2] == "base" for config in sweeps))
        self.assertTrue(all(config[-1] == {
            "population_size": 20, "generations": 40, "mutation_rate": 0.05,
        } for config in baselines))
        self.assertEqual(
            {(config[0], config[5]) for config in sweeps},
            {
                ("population-sweep", 10), ("population-sweep", 20), ("population-sweep", 40),
                ("generation-sweep", 20), ("generation-sweep", 40), ("generation-sweep", 80),
                ("mutation-sweep", 0.0), ("mutation-sweep", 0.05), ("mutation-sweep", 0.2),
            },
        )

    def test_summary_reports_mean_spread_and_successful_packages(self):
        records = [
            {
                "experiment": "baseline",
                "instance": "base",
                "category": "base",
                "factor": "baseline",
                "factor_value": "all",
                "population_size": 20,
                "generations": 40,
                "mutation_rate": 0.05,
                "seed": seed,
                "loaded_packages": loaded,
                "valid": True,
                "evaluations": 820,
                "duration_seconds": duration,
                "result": {"objective_value": value},
            }
            for seed, loaded, duration, value in (
                (11, 6, 1.0, 300),
                (12, 7, 3.0, 320),
            )
        ]

        summary = summarize_ga_analysis(records)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["final_value_mean"], 310)
        self.assertAlmostEqual(summary[0]["final_value_stdev"], 14.1421356)
        self.assertEqual(summary[0]["loaded_packages_mean"], 6.5)
        self.assertEqual(summary[0]["valid_rate"], 1.0)
        self.assertEqual(summary[0]["duration_seconds_mean"], 2)
        self.assertEqual(summary[0]["evaluations_mean"], 820)
        self.assertEqual(summary[0]["factor_value"], "all")


if __name__ == "__main__":
    unittest.main()
