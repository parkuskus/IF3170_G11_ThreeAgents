import json
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from src.visualizer.app import app
    from src.visualizer.recording import MAX_IMPROVEMENTS, MAX_SAMPLES
except ImportError:  # the visualizer dependencies are optional
    TestClient = None

FIXTURE = Path(__file__).parent / "fixtures" / "demo.json"


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class VisualizerApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.problem = {"truck": data["truck"], "packages": data["packages"]}

    def run_search(self, algorithm, **extra):
        return self.client.post(
            "/api/run",
            json={"problem": self.problem, "algorithm": algorithm, "seed": 3, **extra},
        )

    def test_every_algorithm_runs(self):
        for key in ("hill_climbing", "simulated_annealing", "genetic_algorithm"):
            with self.subTest(algorithm=key):
                response = self.run_search(key)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["frames"], [])

    def test_invalid_parameter_is_rejected(self):
        response = self.run_search("genetic_algorithm", parameters={"mutation_rate": 3})
        self.assertEqual(response.status_code, 422)

    def test_animation_frames_start_at_initial_and_end_at_the_final_state(self):
        for key in ("hill_climbing", "simulated_annealing", "genetic_algorithm"):
            with self.subTest(algorithm=key):
                body = self.run_search(key, animate=True).json()
                frames = body["frames"]
                self.assertGreater(len(frames), 1)
                self.assertEqual(frames[0]["placements"], body["initial"]["placements"])
                self.assertEqual(frames[-1]["placements"], body["final"]["placements"])
                evaluations = [frame["evaluation"] for frame in frames[:-1]]
                self.assertEqual(evaluations, sorted(set(evaluations)))
                best = [frame["best"] for frame in frames]
                self.assertEqual(best, sorted(best))

    def test_animation_keeps_a_bounded_number_of_frames(self):
        body = self.run_search(
            "simulated_annealing", animate=True, parameters={"max_iterations": 20000}
        ).json()
        self.assertLessEqual(len(body["frames"]), MAX_SAMPLES + MAX_IMPROVEMENTS + 1)

    def test_animation_does_not_change_the_result(self):
        plain = self.run_search("simulated_annealing").json()
        animated = self.run_search("simulated_annealing", animate=True).json()
        self.assertEqual(plain["objectiveValue"], animated["objectiveValue"])
        self.assertEqual(plain["final"]["placements"], animated["final"]["placements"])


if __name__ == "__main__":
    unittest.main()
