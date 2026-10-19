"""Tests for paired multi-truck algorithm benchmarks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.fleet.benchmark import run_fleet_benchmark
from src.visualization.benchmark import plot_fleet_benchmark_summary


class FleetBenchmarkTests(unittest.TestCase):
    def test_methods_share_initial_states_and_write_valid_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            (fixtures / "tight-space.json").write_text(
                json.dumps(
                    {
                        "packages": [
                            {
                                "id": "p1",
                                "dimensions": [1, 1, 1],
                                "value": 10,
                                "weight": 1,
                                "fragile": False,
                                "eta": 1,
                            }
                        ],
                        "trucks": [
                            {"dimensions": [2, 2, 2], "max_capacity": 2},
                            {"dimensions": [2, 2, 2], "max_capacity": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = run_fleet_benchmark(
                fixtures, seeds=(11, 12, 13), output_dir=root / "output"
            )

            self.assertEqual(len(records), 18)
            for seed in (11, 12, 13):
                seed_records = [row for row in records if row["seed"] == seed]
                self.assertEqual(len(seed_records), 6)
                starts = [row["initial_state"] for row in seed_records]
                self.assertTrue(all(state == starts[0] for state in starts[1:]))
                self.assertTrue(all(row["initial_valid"] for row in seed_records))
                self.assertTrue(all(row["valid"] for row in seed_records))
                self.assertTrue(
                    all(row["initial_state_seed"] == (seed ^ 0x5EED) for row in seed_records)
                )

            output = root / "output"
            self.assertTrue((output / "runs.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "summary.png").is_file())
            representative_states = output / "representative-states"
            self.assertEqual(len(list(representative_states.glob("*.png"))), 6)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["summary"]), 6)
            self.assertTrue(all(row["runs"] == 3 for row in summary["summary"]))
            initial = summary["initial_state_summary"][0]
            self.assertEqual(initial["runs"], 3)
            self.assertEqual(initial["valid_rate"], 1.0)
            self.assertEqual(len(initial["per_truck"]), 2)
            self.assertAlmostEqual(
                sum(row["package_count_mean"] for row in initial["per_truck"]),
                initial["loaded_packages_mean"],
            )

    def test_requires_at_least_three_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                run_fleet_benchmark(root, seeds=(11, 12), output_dir=root / "out")
            with self.assertRaises(ValueError):
                run_fleet_benchmark(root, seeds=(11, 11, 12), output_dir=root / "out")

    def test_summary_plot_allocates_enough_panels_for_custom_fixture_sets(self):
        import matplotlib.pyplot as plt

        summaries = [
            {
                "instance": f"fixture-{index}",
                "method": "ga",
                "final_value_mean": 1,
                "final_value_stdev": 0,
            }
            for index in range(13)
        ]
        original_subplots = plt.subplots
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "src.visualization.benchmark.plt.subplots", wraps=original_subplots
            ) as create_subplots:
                plot_fleet_benchmark_summary(summaries, Path(temporary) / "plot.png")
        self.assertEqual(create_subplots.call_args.args[:2], (4, 4))


if __name__ == "__main__":
    unittest.main()
