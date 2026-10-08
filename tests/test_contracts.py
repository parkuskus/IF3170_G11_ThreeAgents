import unittest

from src.contracts import resolve_objective
from src.model import Package, Placement, State, Truck


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


if __name__ == "__main__":
    unittest.main()
