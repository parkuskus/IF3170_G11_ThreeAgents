"""Constraint checks for a single-truck loading state."""

from itertools import combinations

from src.model import Package, Placement, State


def _overlap_1d(start_a: int, size_a: int, start_b: int, size_b: int) -> bool:
    return start_a < start_b + size_b and start_b < start_a + size_a


def _overlap_xy(a: Placement, b: Placement) -> bool:
    return _overlap_1d(a.coordinate[0], a.oriented_dimensions[0], b.coordinate[0], b.oriented_dimensions[0]) and _overlap_1d(
        a.coordinate[1], a.oriented_dimensions[1], b.coordinate[1], b.oriented_dimensions[1]
    )


def _overlap_3d(a: Placement, b: Placement) -> bool:
    return _overlap_xy(a, b) and _overlap_1d(
        a.coordinate[2], a.oriented_dimensions[2], b.coordinate[2], b.oriented_dimensions[2]
    )


def validate_state(state: State) -> list[str]:
    """Return all constraint violations. An empty list means the state is valid."""
    errors = []
    placed: list[tuple[Package, Placement]] = [
        (package, placement)
        for package in state.packages
        if (placement := state.placements[package.id]) is not None
    ]

    for package, placement in placed:
        if any(
            position + size > limit
            for position, size, limit in zip(
                placement.coordinate, placement.oriented_dimensions, state.truck.dimensions
            )
        ):
            errors.append(f"package {package.id!r} is outside the truck")

    for (package_a, placement_a), (package_b, placement_b) in combinations(placed, 2):
        if _overlap_3d(placement_a, placement_b):
            errors.append(f"packages {package_a.id!r} and {package_b.id!r} overlap")

    total_weight = sum(package.weight for package, _ in placed)
    if total_weight > state.truck.max_capacity:
        errors.append("truck capacity is exceeded")

    for package, placement in placed:
        if placement.coordinate[2] == 0:
            continue
        supported = any(
            not support_package.fragile
            and support_placement.coordinate[2] + support_placement.oriented_dimensions[2]
            == placement.coordinate[2]
            and _overlap_xy(placement, support_placement)
            for support_package, support_placement in placed
            if support_package.id != package.id
        )
        if not supported:
            errors.append(f"package {package.id!r} is unsupported")

    return errors
