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

    def fleet_problem(self, trucks=2):
        return {"trucks": [self.problem["truck"]] * trucks, "packages": self.problem["packages"]}

    def test_single_truck_result_is_a_fleet_of_one(self):
        body = self.run_search("hill_climbing").json()
        self.assertEqual(len(body["trucks"]), 1)
        self.assertEqual(len(body["final"]["placements"]), 1)
        self.assertEqual(len(body["final"]["placedWeight"]), 1)

    def test_every_algorithm_runs_on_a_fleet(self):
        for key in ("hill_climbing", "simulated_annealing", "genetic_algorithm"):
            with self.subTest(algorithm=key):
                response = self.client.post(
                    "/api/run",
                    json={"problem": self.fleet_problem(3), "algorithm": key, "seed": 3, "animate": True},
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(len(body["trucks"]), 3)
                for state in (body["initial"], body["final"]):
                    self.assertEqual(len(state["placements"]), 3)
                    self.assertEqual(state["violations"], [])
                    loaded = [
                        package_id
                        for truck in state["placements"]
                        for package_id, placement in truck.items()
                        if placement is not None
                    ]
                    self.assertEqual(len(loaded), len(set(loaded)))  # no package in two trucks
                self.assertEqual(body["frames"][-1]["placements"], body["final"]["placements"])

    def test_more_trucks_never_hurt_total_value(self):
        def best(trucks):
            response = self.client.post(
                "/api/run",
                json={
                    "problem": self.fleet_problem(trucks),
                    "algorithm": "genetic_algorithm",
                    "seed": 1,
                    "parameters": {"population_size": 40, "generations": 60},
                },
            )
            return response.json()["objectiveValue"]

        self.assertGreaterEqual(best(3), best(1))

    def test_examples_include_fleet_variants(self):
        examples = {e["name"]: e["problem"] for e in self.client.get("/api/examples").json()}
        self.assertEqual(len(examples["base-2-trucks"]["trucks"]), 2)
        self.assertEqual(len(examples["demo"]["trucks"]), 1)

    def test_truck_and_trucks_together_are_rejected(self):
        problem = {**self.problem, "trucks": [self.problem["truck"]]}
        response = self.client.post("/api/run", json={"problem": problem, "algorithm": "hill_climbing"})
        self.assertEqual(response.status_code, 422)

    def test_animation_does_not_change_the_result(self):
        plain = self.run_search("simulated_annealing").json()
        animated = self.run_search("simulated_annealing", animate=True).json()
        self.assertEqual(plain["objectiveValue"], animated["objectiveValue"])
        self.assertEqual(plain["final"]["placements"], animated["final"]["placements"])


if __name__ == "__main__":
    unittest.main()
