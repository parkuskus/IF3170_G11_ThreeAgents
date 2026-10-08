"""Shared domain model for the single-truck package-loading problem."""

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping, TypeAlias


Dimensions: TypeAlias = tuple[int, int, int]
Coordinate: TypeAlias = tuple[int, int, int]
Orientation: TypeAlias = tuple[int, int, int]

ORIENTATIONS: tuple[Orientation, ...] = tuple(permutations((0, 1, 2)))


def _positive_dimensions(dimensions: Dimensions, name: str) -> None:
    if len(dimensions) != 3 or any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError(f"{name} must contain three positive integers")


@dataclass(frozen=True, slots=True)
class Package:
    id: str
    dimensions: Dimensions
    value: int
    weight: int
    fragile: bool
    eta: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("package id must not be empty")
        _positive_dimensions(self.dimensions, "package dimensions")
        if type(self.value) is not int or self.value < 0:
            raise ValueError("package value must be a non-negative integer")
        if type(self.weight) is not int or self.weight < 0:
            raise ValueError("package weight must be a non-negative integer")
        if type(self.fragile) is not bool:
            raise ValueError("package fragile must be a bool")
        if type(self.eta) is not int or self.eta < 0:
            raise ValueError("package ETA must be a non-negative integer")

    def oriented_dimensions(self, orientation: Orientation = (0, 1, 2)) -> Dimensions:
        if len(orientation) != 3 or set(orientation) != {0, 1, 2}:
            raise ValueError("orientation must be a permutation of (0, 1, 2)")
        return tuple(self.dimensions[index] for index in orientation)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Truck:
    dimensions: Dimensions
    max_capacity: int

    def __post_init__(self) -> None:
        _positive_dimensions(self.dimensions, "truck dimensions")
        if type(self.max_capacity) is not int or self.max_capacity < 0:
            raise ValueError("truck capacity must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class Placement:
    """Position and orientation of a package inside the truck."""

    coordinate: Coordinate
    oriented_dimensions: Dimensions
    orientation: Orientation = (0, 1, 2)

    def __post_init__(self) -> None:
        if len(self.coordinate) != 3 or any(type(value) is not int or value < 0 for value in self.coordinate):
            raise ValueError("placement coordinate must contain three non-negative integers")
        _positive_dimensions(self.oriented_dimensions, "oriented dimensions")
        if len(self.orientation) != 3 or set(self.orientation) != {0, 1, 2}:
            raise ValueError("orientation must be a permutation of (0, 1, 2)")


@dataclass(frozen=True, slots=True)
class State:
    """A single-truck state; None means that package is outside the truck."""

    packages: tuple[Package, ...]
    truck: Truck
    placements: Mapping[str, Placement | None]

    def __post_init__(self) -> None:
        package_ids = [package.id for package in self.packages]
        if len(package_ids) != len(set(package_ids)):
            raise ValueError("package IDs must be unique")
        if set(self.placements) != set(package_ids):
            raise ValueError("placements must contain exactly one entry for each package ID")
        packages_by_id = {package.id: package for package in self.packages}
        for package_id, placement in self.placements.items():
            if placement is not None and placement.oriented_dimensions != packages_by_id[
                package_id
            ].oriented_dimensions(placement.orientation):
                raise ValueError(f"oriented dimensions do not match package {package_id!r}")

    def objective(self) -> int:
        """Return the total value of packages currently placed in the truck."""
        return sum(
            package.value
            for package in self.packages
            if self.placements[package.id] is not None
        )
