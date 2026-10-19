import unittest
from pathlib import Path

from src.experiments.common.benchmark import discover_instances
from src.experiments.sa.results_analysis import (
    BASELINE,
    _configurations,
    _iteration_of_best,
    _run_directory,
    _worse_moves_accepted,
    summarize_sa_analysis,
)


class SAResultsAnalysisTests(unittest.TestCase):
    def test_each_sweep_configuration_gets_its_own_run_directory(self):
        configurations = _configurations(discover_instances(Path("tests/fixtures")))
        directories = {
            _run_directory(Path("output"), experiment, instance, factor, value, 11)
            for experiment, _, instance, _, factor, value, _ in configurations
        }

        self.assertEqual(len(directories), 20)

    def test_protocol_covers_all_fixture_baselines_and_three_base_sweeps(self):
        configurations = list(_configurations(discover_instances(Path("tests/fixtures"))))
        baselines = [config for config in configurations if config[0] == "baseline"]
        sweeps = [config for config in configurations if config[0] != "baseline"]

        self.assertEqual(len(baselines), 11)
        self.assertEqual(len(sweeps), 9)
        self.assertTrue(all(config[2] == "base" for config in sweeps))
        self.assertTrue(all(config[-1] == BASELINE for config in baselines))
        self.assertEqual(
            {(config[0], config[5]) for config in sweeps},
            {
                ("temperature-sweep", 10.0), ("temperature-sweep", 100.0), ("temperature-sweep", 1000.0),
                ("cooling-sweep", 0.99), ("cooling-sweep", 0.995), ("cooling-sweep", 0.999),
                ("iteration-sweep", 1000), ("iteration-sweep", 3000), ("iteration-sweep", 6000),
            },
        )

    def test_dynamics_count_worse_moves_and_first_best_iteration(self):
        self.assertEqual(_worse_moves_accepted((10, 8, 8, 12, 9, 15)), 2)
        self.assertEqual(_iteration_of_best((10, 10, 12, 15, 15)), 3)
        self.assertEqual(_iteration_of_best((10, 10)), 0)

    def test_summary_reports_mean_spread_and_dynamics(self):
        records = [
            {
                "experiment": "baseline",
                "instance": "base",
                "factor": "baseline",
                "factor_value": "all",
                "seed": seed,
                "initial_value": initial,
                "loaded_packages": loaded,
                "valid": True,
                "worse_accepted": worse,
                "best_iteration": best_iteration,
                "stuck_count": 1,
                "final_temperature": 0.5,
                "evaluations": 3001,
                "duration_seconds": duration,
                "result": {"objective_value": value},
            }
            for seed, initial, loaded, worse, best_iteration, duration, value in (
                (11, 280, 6, 4, 100, 1.0, 300),
                (12, 290, 7, 6, 300, 3.0, 320),
            )
        ]

        summary = summarize_sa_analysis(records)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["initial_value_mean"], 285)
        self.assertEqual(summary[0]["final_value_mean"], 310)
        self.assertAlmostEqual(summary[0]["final_value_stdev"], 14.1421356)
        self.assertEqual(summary[0]["worse_accepted_mean"], 5)
        self.assertEqual(summary[0]["best_iteration_mean"], 200)
        self.assertEqual(summary[0]["valid_rate"], 1.0)
        self.assertEqual(summary[0]["factor_value"], "all")


if __name__ == "__main__":
    unittest.main()
