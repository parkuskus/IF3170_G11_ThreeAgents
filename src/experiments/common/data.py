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


def _load_packages(data: Any) -> tuple[Package, ...]:
    if not isinstance(data, dict) or not isinstance(data.get("packages"), list):
        raise ValueError("problem JSON must contain a packages list")
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
    return tuple(packages)


def _load_truck(truck_data: Any, name: str) -> Truck:
    if not isinstance(truck_data, dict):
        raise ValueError(f"{name} must be an object")
    try:
        return Truck(
            dimensions=_dimensions(truck_data["dimensions"], f"{name}.dimensions"),
            max_capacity=truck_data["max_capacity"],
        )
    except KeyError as error:
        raise ValueError(f"{name} is missing {error.args[0]!r}") from error


def load_problem(path: str | Path) -> tuple[tuple[Package, ...], Truck]:
    """Load package and single-truck data from the documented JSON shape."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    packages = _load_packages(data)
    if "truck" not in data:
        raise ValueError("problem JSON must contain a truck object")
    return packages, _load_truck(data["truck"], "truck")


def load_fleet_problem(
    path: str | Path, *, truck_count: int | None = None
) -> tuple[tuple[Package, ...], tuple[Truck, ...]]:
    """Load an explicit heterogeneous fleet or replicate a legacy truck."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    packages = _load_packages(data)
    if "trucks" in data:
        if truck_count is not None:
            raise ValueError("truck_count cannot be used with an explicit trucks list")
        truck_data = data["trucks"]
        if not isinstance(truck_data, list) or not truck_data:
            raise ValueError("trucks must be a non-empty list")
        trucks = tuple(
            _load_truck(item, f"trucks[{index}]") for index, item in enumerate(truck_data)
        )
        return packages, trucks

    if "truck" not in data:
        raise ValueError("problem JSON must contain a truck object or trucks list")
    if truck_count is None:
        truck_count = 1
    if type(truck_count) is not int or truck_count < 1:
        raise ValueError("truck_count must be a positive integer")
    truck = _load_truck(data["truck"], "truck")
    return packages, (truck,) * truck_count
