import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from src.algorithms.hill_climbing import VARIANTS, hill_climbing
from src.contracts import SearchProblem
from src.experiments.common.data import load_problem
from src.experiments.hc.experiments import run_hc_experiments
from src.model import FleetState
from src.neighbors import random_initial_state
from src.results import HillClimbingMetrics
from src.search_problems import search_problem_for_state
from src.validation import validate_fleet_state, validate_state

DEMO = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.json"


class HillClimbingTests(unittest.TestCase):
    def setUp(self):
        self.packages, self.truck = load_problem(DEMO)
        self.start = random_initial_state(self.packages, self.truck, Random(3))

    def run_variant(self, variant, **kw):
        return hill_climbing(self.start, seed=5, variant=variant, max_restarts=3, **kw)

    def test_every_variant_returns_valid_non_worsening_result(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                r = self.run_variant(variant)
                self.assertEqual(validate_state(r.final_state), [])
                self.assertGreaterEqual(r.objective_value, self.start.objective())
                self.assertEqual(r.objective_value, r.final_state.objective())
                self.assertIsInstance(r.metrics, HillClimbingMetrics)
                self.assertIs(r.initial_state, self.start)
                self.assertEqual(r.seed, 5)

    def test_same_seed_is_reproducible(self):
        for variant in VARIANTS:
            a, b = self.run_variant(variant), self.run_variant(variant)
            self.assertEqual(a.objective_history, b.objective_history)

    def test_history_never_decreases_within_a_climb(self):
        for variant in ("steepest", "sideways", "stochastic"):
            h = self.run_variant(variant).objective_history
            self.assertTrue(all(x <= y for x, y in zip(h, h[1:])))

    def test_sideways_respects_budget(self):
        r = self.run_variant("sideways", max_sideways=7)
        self.assertLessEqual(r.metrics.sideways_moves, 7)
        self.assertEqual(r.metrics.maximum_sideways_moves, 7)

    def test_random_restart_metrics(self):
        r = self.run_variant("random_restart")
        self.assertEqual(r.metrics.restarts, 3)
        self.assertEqual(len(r.metrics.iterations_per_restart), 4)
        self.assertEqual(r.metrics.iterations, sum(r.metrics.iterations_per_restart))
        self.assertEqual(r.objective_value, max(r.objective_history))

    def test_stochastic_uses_full_iteration_budget(self):
        r = hill_climbing(self.start, seed=1, variant="stochastic", max_iterations=25)
        self.assertEqual(r.metrics.iterations, 25)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            hill_climbing(self.start, seed=1, variant="nope")
        with self.assertRaises(ValueError):
            hill_climbing(self.start, seed=1, neighbors_per_iteration=0)

    def test_hill_climbing_accepts_fleet_search_problem(self):
        fleet = FleetState(
            self.packages,
            (self.truck, self.truck),
            tuple({package.id: None for package in self.packages} for _ in range(2)),
        )
        problem = search_problem_for_state(fleet)
        for variant in ("steepest", "random_restart"):
            result = hill_climbing(
                fleet,
                seed=7,
                variant=variant,
                max_iterations=3,
                max_restarts=1,
                search_problem=problem,
            )
            self.assertEqual(validate_fleet_state(result.final_state), [])

    def test_hill_climbing_rejects_invalid_problem_neighbor(self):
        problem = SearchProblem(
            objective_fn=lambda state: state,
            is_valid=lambda state: state >= 0,
            random_state=lambda rng: 0,
            neighbor=lambda state, rng: -1,
        )
        with self.assertRaisesRegex(ValueError, "neighbor.*invalid"):
            hill_climbing(0, seed=1, search_problem=problem, max_iterations=1)

    def test_experiment_runner_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = run_hc_experiments(
                self.packages, self.truck, seeds=(1, 2, 3), output_dir=tmp,
                max_iterations=20, max_restarts=2,
            )
            self.assertEqual(len(records), 12)
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual([s["variant"] for s in summary], list(VARIANTS))
            self.assertTrue((Path(tmp) / "steepest" / "run-1" / "objective-history.png").exists())
            self.assertTrue((Path(tmp) / "summary.png").exists())


if __name__ == "__main__":
    unittest.main()
