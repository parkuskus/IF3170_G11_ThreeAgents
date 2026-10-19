import unittest
from random import Random

from src.contracts import SearchProblem, resolve_objective
from src.model import FleetState, Package, Placement, State, Truck
from src.search_problems import search_problem_for_state
from src.validation import validate_fleet_state, validate_state


class AlgorithmContractTests(unittest.TestCase):
    def setUp(self):
        self.state = State(
            (Package("box", (1, 1, 1), 10, 1, False, 1),),
            Truck((2, 2, 2), 5),
            {"box": Placement((0, 0, 0), (1, 1, 1))},
        )

    def test_default_objective_uses_state_objective(self):
        self.assertEqual(resolve_objective(self.state)(self.state), 10)

    def test_custom_objective_overrides_default(self):
        objective = resolve_objective(self.state, lambda state: state.objective() * 2)
        self.assertEqual(objective(self.state), 20)

    def test_custom_objective_supports_state_without_method(self):
        objective = resolve_objective({"value": 7}, lambda state: state["value"])
        self.assertEqual(objective({"value": 7}), 7)

    def test_missing_objective_is_rejected(self):
        with self.assertRaises(TypeError):
            resolve_objective({"value": 7})

    def test_search_problem_contract_supports_arbitrary_state_type(self):
        problem = SearchProblem(
            objective_fn=lambda state: state["score"],
            is_valid=lambda state: state["score"] >= 0,
            random_state=lambda rng: {"score": rng.randrange(10)},
            neighbor=lambda state, rng: {"score": state["score"] + 1},
        )
        state = problem.random_state(Random(3))
        self.assertTrue(problem.is_valid(state))
        self.assertEqual(problem.objective_fn(state), state["score"])
        self.assertEqual(problem.neighbor(state, Random(1))["score"], state["score"] + 1)
        self.assertIsNone(problem.crossover)
        self.assertIsNone(problem.mutate)

    def test_default_search_problem_adapts_single_truck_state(self):
        problem = search_problem_for_state(self.state)
        self.assertEqual(problem.objective_fn(self.state), 10)
        self.assertTrue(problem.is_valid(self.state))
        self.assertTrue(validate_state(problem.random_state(Random(5))) == [])
        self.assertTrue(validate_state(problem.neighbor(self.state, Random(5))) == [])
        self.assertIsNotNone(problem.crossover)
        self.assertIsNotNone(problem.mutate)

    def test_default_search_problem_adapts_fleet_state(self):
        package = self.state.packages[0]
        fleet = FleetState(
            (package,),
            (self.state.truck, Truck((1, 1, 1), 1)),
            ({"box": self.state.placements["box"]}, {"box": None}),
        )
        problem = search_problem_for_state(fleet)
        self.assertEqual(problem.objective_fn(fleet), 10)
        self.assertTrue(problem.is_valid(fleet))
        self.assertEqual(validate_fleet_state(problem.random_state(Random(5))), [])
        self.assertEqual(validate_fleet_state(problem.neighbor(fleet, Random(5))), [])
        self.assertIsNotNone(problem.crossover)
        self.assertIsNotNone(problem.mutate)

    def test_unknown_state_requires_explicit_search_problem(self):
        with self.assertRaisesRegex(TypeError, "SearchProblem"):
            search_problem_for_state({"score": 7})


if __name__ == "__main__":
    unittest.main()
