import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.fleet.benchmark import (
    _expand_identical_trucks,
    run_fleet_benchmark,
    summarize_fleet_benchmark,
    summarize_initial_fleet_states,
)
from src.model import Truck


class FleetBenchmarkTests(unittest.TestCase):
    def test_expands_identical_truck_specification_to_requested_count(self):
        truck = Truck(dimensions=(8, 6, 4), max_capacity=30)

        expanded = _expand_identical_trucks((truck, truck), 5)

        self.assertEqual(expanded, (truck, truck, truck, truck, truck))

    def test_rejects_heterogeneous_truck_specifications(self):
        trucks = (
            Truck(dimensions=(8, 6, 4), max_capacity=30),
            Truck(dimensions=(6, 4, 3), max_capacity=30),
        )

        with self.assertRaisesRegex(ValueError, "identical"):
            _expand_identical_trucks(trucks, 3)

    def test_summary_keeps_truck_counts_separate(self):
        records = [
            {
                "truck_count": count,
                "category": "base",
                "instance": "base",
                "method": "sa",
                "seed": 11,
                "initial_state": {
                    "truck_count": count,
                    "total_package_value": 10,
                    "trucks": [
                        {"packages": [], "total_package_value": 0}
                        for _ in range(count)
                    ],
                    "unloaded_packages": ["p"],
                },
                "initial_valid": True,
                "initial_value": 10,
                "initial_loaded_packages": 0,
                "final_value": 20,
                "final_loaded_packages": 1,
                "valid": True,
                "objective_evaluations": 5,
                "duration_seconds": 0.1,
            }
            for count in (3, 5)
        ]

        results = summarize_fleet_benchmark(records)
        initial_states = summarize_initial_fleet_states(records)

        self.assertEqual([row["truck_count"] for row in results], [3, 5])
        self.assertEqual([row["truck_count"] for row in initial_states], [3, 5])
        self.assertEqual([len(row["per_truck"]) for row in initial_states], [3, 5])

    def test_benchmark_saves_count_specific_summary_and_state_visuals(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            (fixtures / "tight-space.json").write_text(
                json.dumps(
                    {
                        "packages": [
                            {
                                "id": "p",
                                "dimensions": [1, 1, 1],
                                "value": 10,
                                "weight": 1,
                                "fragile": False,
                                "eta": 0,
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
            (fixtures / "base.json").write_text(
                (fixtures / "tight-space.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            output = root / "output"

            with (
                patch("src.experiments.fleet.benchmark.HC_PARAMETERS", {
                    "neighbors_per_iteration": 1,
                    "max_iterations": 1,
                    "max_sideways": 1,
                    "max_restarts": 1,
                }),
                patch("src.experiments.fleet.benchmark.SA_PARAMETERS", {
                    "max_iterations": 1,
                    "initial_temperature": 100.0,
                    "cooling_rate": 0.995,
                    "stuck_patience": 1,
                }),
                patch("src.experiments.fleet.benchmark.GA_PARAMETERS", {
                    "population_size": 2,
                    "generations": 1,
                    "mutation_rate": 0.05,
                }),
            ):
                records = run_fleet_benchmark(
                    fixtures,
                    seeds=(11,),
                    truck_counts=(3,),
                    instance_names=("tight-space",),
                    output_dir=output,
                )

            self.assertEqual(len(records), 6)
            self.assertEqual({record["instance"] for record in records}, {"tight-space"})
            self.assertEqual({record["truck_count"] for record in records}, {3})
            self.assertEqual({record["seed"] for record in records}, {11})
            self.assertTrue(all(len(record["final_state"]["trucks"]) == 3 for record in records))
            self.assertTrue((output / "summary-trucks-3.png").is_file())
            visualizations = list((output / "representative-states" / "trucks-3").glob("*.png"))
            self.assertEqual(len(visualizations), 6)


if __name__ == "__main__":
    unittest.main()
