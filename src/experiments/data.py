"""JSON input loader for package-loading experiments."""

import json
from pathlib import Path
from typing import Any

from src.model import Package, Truck


def _dimensions(value: Any, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(number) is not int for number in value)
    ):
        raise ValueError(f"{name} must be a list of three integers")
    return tuple(value)


def load_problem(path: str | Path) -> tuple[tuple[Package, ...], Truck]:
    """Load package and truck data from the documented JSON shape."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("packages"), list):
        raise ValueError("problem JSON must contain a packages list")
    truck_data = data.get("truck")
    if not isinstance(truck_data, dict):
        raise ValueError("problem JSON must contain a truck object")
    if not data["packages"]:
        raise ValueError("problem must contain at least one package")

    packages = []
    for index, package_data in enumerate(data["packages"]):
        if not isinstance(package_data, dict):
            raise ValueError(f"packages[{index}] must be an object")
        try:
            packages.append(
                Package(
                    id=package_data["id"],
                    dimensions=_dimensions(
                        package_data["dimensions"], f"packages[{index}].dimensions"
                    ),
                    value=package_data["value"],
                    weight=package_data["weight"],
                    fragile=package_data["fragile"],
                    eta=package_data["eta"],
                )
            )
        except KeyError as error:
            raise ValueError(f"packages[{index}] is missing {error.args[0]!r}") from error

    if len({package.id for package in packages}) != len(packages):
        raise ValueError("package IDs must be unique")
    try:
        truck = Truck(
            dimensions=_dimensions(truck_data["dimensions"], "truck.dimensions"),
            max_capacity=truck_data["max_capacity"],
        )
    except KeyError as error:
        raise ValueError(f"truck is missing {error.args[0]!r}") from error
    return tuple(packages), truck
