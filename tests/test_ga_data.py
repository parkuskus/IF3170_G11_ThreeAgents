import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.experiments.data import load_problem
from src.experiments.run_ga import main


class GADataTests(unittest.TestCase):
    def test_loads_demo_problem(self):
        path = Path(__file__).parents[1] / "examples" / "ga-demo.json"
        packages, truck = load_problem(path)
        self.assertEqual(len(packages), 5)
        self.assertEqual(truck.max_capacity, 12)

    def test_rejects_invalid_problem_shape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "problem.json"
            path.write_text(json.dumps({"packages": [], "truck": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_problem(path)

    def test_cli_loads_problem_and_dispatches_runner(self):
        problem = Path(__file__).parents[1] / "examples" / "ga-demo.json"
        with patch("src.experiments.run_ga.run_ga_experiments", return_value=()) as runner:
            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--input", str(problem),
                        "--generation-values", "4", "6", "8",
                        "--population-values", "4", "6", "8",
                        "--fixed-population-size", "6",
                        "--fixed-generations", "6",
                        "--seeds", "41", "42", "43",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(runner.call_count, 1)


if __name__ == "__main__":
    unittest.main()
