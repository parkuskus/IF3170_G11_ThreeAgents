"""Tests for paired multi-truck algorithm benchmarks."""

import json
import tempfile
import unittest
from copy import deepcopy
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
                fixtures,
                seeds=(11, 12, 13),
                truck_counts=(2,),
                instance_names=("tight-space",),
                output_dir=root / "output",
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
            self.assertTrue((output / "summary-trucks-2.png").is_file())
            representative_states = output / "representative-states" / "trucks-2"
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

    def test_accepts_one_seed_and_rejects_empty_or_duplicate_seed_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                run_fleet_benchmark(root, seeds=(), output_dir=root / "out")
            with self.assertRaises(ValueError):
                run_fleet_benchmark(root, seeds=(11, 11, 12), output_dir=root / "out")

    def test_rejects_unknown_instance_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            (fixtures / "base.json").write_text(
                json.dumps({
                    "packages": [
                        {
                            "id": "p",
                            "dimensions": [1, 1, 1],
                            "value": 1,
                            "weight": 1,
                            "fragile": False,
                            "eta": 0,
                        }
                    ],
                    "trucks": [
                        {"dimensions": [2, 2, 2], "max_capacity": 2},
                        {"dimensions": [2, 2, 2], "max_capacity": 2},
                    ],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fleet fixtures"):
                run_fleet_benchmark(
                    fixtures,
                    seeds=(11,),
                    truck_counts=(3,),
                    instance_names=("missing",),
                    output_dir=root / "out",
                )

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

    def test_single_run_summary_plot_does_not_show_zero_spread_as_error_bars(self):
        import matplotlib.axes

        summary = {
            "instance": "base",
            "method": "ga",
            "runs": 1,
            "final_value_mean": 600,
            "final_value_stdev": 0,
        }
        original_bar = matplotlib.axes.Axes.bar
        bar_kwargs = []

        def capture_bar(axis, *args, **kwargs):
            bar_kwargs.append(kwargs)
            return original_bar(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(matplotlib.axes.Axes, "bar", capture_bar):
                plot_fleet_benchmark_summary([summary], Path(temporary) / "plot.png")
        self.assertNotIn("yerr", bar_kwargs[0])

    def test_report_transfer_figure_uses_actual_cross_truck_exchange(self):
        from src.visualization.benchmark import plot_fleet_transfer

        records = json.loads(
            (Path(__file__).resolve().parents[3] / "output" / "fleet-count-swap-analysis" / "runs.json")
            .read_text(encoding="utf-8")
        )
        record = next(
            row for row in records
            if (row["instance"], row["method"], row["truck_count"], row["seed"])
            == ("tight-capacity", "hc_sideways", 3, 11)
        )
        assignments = [
            {package["id"]: index + 1
             for index, truck in enumerate(record[side]["trucks"])
             for package in truck["packages"]}
            for side in ("initial_state", "final_state")
        ]
        self.assertEqual(record["initial_value"], record["final_value"])
        self.assertEqual(record["final_value"], 445)
        self.assertEqual((assignments[0]["P03"], assignments[1]["P03"]), (1, 3))
        self.assertEqual((assignments[0]["P04"], assignments[1]["P04"]), (3, 1))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "transfer.png"
            plot_fleet_transfer(record, destination)
            self.assertGreater(destination.stat().st_size, 1000)
            incomplete = deepcopy(record)
            incomplete["final_state"]["trucks"][0]["packages"] = [
                package for package in incomplete["final_state"]["trucks"][0]["packages"]
                if package["id"] != "P06"
            ]
            with self.assertRaisesRegex(ValueError, "reciprocal cross-truck exchange"):
                plot_fleet_transfer(incomplete, destination)


if __name__ == "__main__":
    unittest.main()
