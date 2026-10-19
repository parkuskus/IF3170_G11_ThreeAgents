"""Request models and conversion between JSON and domain objects."""

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

from src.model import Package, Truck
from src.results import SearchResult
from src.validation import validate_state


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
    truck: TruckIn
    packages: list[PackageIn]


class RunRequest(_Strict):
    problem: ProblemIn
    algorithm: str
    objective: str = "total_value"
    seed: StrictInt = 0
    parameters: dict[str, Any] = {}
    # Record the best-so-far states seen during the search for the UI to replay.
    animate: StrictBool = False


def build_problem(problem: ProblemIn) -> tuple[tuple[Package, ...], Truck]:
    """Build domain objects; raises ValueError on any domain-rule violation."""
    if not problem.packages:
        raise ValueError("problem must contain at least one package")
    packages = tuple(
        Package(p.id, tuple(p.dimensions), p.value, p.weight, p.fragile, p.eta)
        for p in problem.packages
    )
    if len({p.id for p in packages}) != len(packages):
        raise ValueError("package IDs must be unique")
    truck = Truck(tuple(problem.truck.dimensions), problem.truck.max_capacity)
    return packages, truck


def placements_json(state) -> dict[str, dict[str, Any] | None]:
    return {
        package_id: (
            None
            if placement is None
            else {
                "coordinate": list(placement.coordinate),
                "dimensions": list(placement.oriented_dimensions),
                "orientation": list(placement.orientation),
            }
        )
        for package_id, placement in state.placements.items()
    }


def _state_json(state) -> dict[str, Any]:
    return {
        "placements": placements_json(state),
        "placedWeight": sum(
            p.weight for p in state.packages if state.placements[p.id] is not None
        ),
        "violations": validate_state(state),
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
