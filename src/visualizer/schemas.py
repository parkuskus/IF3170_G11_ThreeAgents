"""Request models and conversion between JSON and domain objects."""

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, model_validator

from src.model import FleetState, Package, State, Truck
from src.results import SearchResult
from src.validation import validate_fleet_state, validate_state


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PackageIn(_Strict):
    id: StrictStr
    dimensions: list[StrictInt]
    value: StrictInt
    weight: StrictInt
    fragile: StrictBool
    eta: StrictInt


class TruckIn(_Strict):
    dimensions: list[StrictInt]
    max_capacity: StrictInt


class ProblemIn(BaseModel):
    # Fixture metadata such as "description" is tolerated and ignored.
    trucks: list[TruckIn]
    packages: list[PackageIn]

    @model_validator(mode="before")
    @classmethod
    def _single_truck_shorthand(cls, data: Any) -> Any:
        """Accept the single-truck `truck` key, as in the fixture files."""
        if isinstance(data, dict) and "truck" in data:
            if "trucks" in data:
                raise ValueError("give either truck or trucks, not both")
            data = {**data, "trucks": [data["truck"]]}
            del data["truck"]
        return data


class RunRequest(_Strict):
    problem: ProblemIn
    algorithm: str
    objective: str = "total_value"
    seed: StrictInt = 0
    parameters: dict[str, Any] = {}
    # Record the best-so-far states seen during the search for the UI to replay.
    animate: StrictBool = False


def build_problem(problem: ProblemIn) -> tuple[tuple[Package, ...], tuple[Truck, ...]]:
    """Build domain objects; raises ValueError on any domain-rule violation."""
    if not problem.packages:
        raise ValueError("problem must contain at least one package")
    if not problem.trucks:
        raise ValueError("problem must contain at least one truck")
    packages = tuple(
        Package(p.id, tuple(p.dimensions), p.value, p.weight, p.fragile, p.eta)
        for p in problem.packages
    )
    if len({p.id for p in packages}) != len(packages):
        raise ValueError("package IDs must be unique")
    trucks = tuple(Truck(tuple(t.dimensions), t.max_capacity) for t in problem.trucks)
    return packages, trucks


def _truck_maps(state: State | FleetState):
    """One placement map per truck; a single-truck State is a fleet of one."""
    return (state.placements,) if isinstance(state, State) else state.placements


def placements_json(state: State | FleetState) -> list[dict[str, dict[str, Any] | None]]:
    return [
        {
            package_id: (
                None
                if placement is None
                else {
                    "coordinate": list(placement.coordinate),
                    "dimensions": list(placement.oriented_dimensions),
                    "orientation": list(placement.orientation),
                }
            )
            for package_id, placement in truck_placements.items()
        }
        for truck_placements in _truck_maps(state)
    ]


def _state_json(state: State | FleetState) -> dict[str, Any]:
    violations = (
        validate_state(state) if isinstance(state, State) else validate_fleet_state(state)
    )
    return {
        "placements": placements_json(state),
        "placedWeight": [
            sum(p.weight for p in state.packages if truck_placements[p.id] is not None)
            for truck_placements in _truck_maps(state)
        ],
        "violations": violations,
    }


def result_json(result: SearchResult) -> dict[str, Any]:
    return {
        "algorithm": result.algorithm,
        "objectiveName": result.objective_name,
        "objectiveValue": result.objective_value,
        "objectiveHistory": list(result.objective_history),
        "durationSeconds": result.duration_seconds,
        "seed": result.seed,
        "parameters": dict(result.parameters),
        "metrics": asdict(result.metrics),
        "initial": _state_json(result.initial_state),
        "final": _state_json(result.final_state),
    }
