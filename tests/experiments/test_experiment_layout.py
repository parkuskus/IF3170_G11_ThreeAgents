import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ExperimentLayoutTests(unittest.TestCase):
    def test_experiment_modules_are_grouped_and_cli_entrypoints_remain(self):
        for path in (
            "src/experiments/common/data.py",
            "src/experiments/ga/experiments.py",
            "src/experiments/hc/experiments.py",
            "src/experiments/sa/experiments.py",
            "src/experiments/fleet/ga.py",
            "src/experiments/run_fleet_ga.py",
        ):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

    def test_tests_are_grouped_into_discoverable_packages(self):
        for folder in ("core", "algorithms", "experiments", "fleet"):
            with self.subTest(folder=folder):
                self.assertTrue((ROOT / "tests" / folder / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
