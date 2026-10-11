"""Shared helper for proposing one valid package placement."""

from random import Random

from src.model import ORIENTATIONS, Coordinate, Orientation, Package, Placement, State
from src.validation import validate_state


def random_valid_placement(
    state: State,
    package: Package,
    rng: Random,
    *,
    orientation: Orientation | None = None,
    coordinate: Coordinate | None = None,
    attempts: int = 50,
) -> Placement | None:
    """Find a random placement for one gene that keeps the whole state valid."""
    if attempts < 1:
        return None
    orientations = (orientation,) if orientation is not None else ORIENTATIONS
    others = {
        package_id: placement
        for package_id, placement in state.placements.items()
        if package_id != package.id and placement is not None
    }
    packages_by_id = {item.id: item for item in state.packages}
    possible_z = {0}
    possible_z.update(
        placement.coordinate[2] + placement.oriented_dimensions[2]
        for package_id, placement in others.items()
        if not packages_by_id[package_id].fragile
    )

    for _ in range(attempts):
        selected_orientation = rng.choice(orientations)
        dimensions = package.oriented_dimensions(selected_orientation)
        if any(size > limit for size, limit in zip(dimensions, state.truck.dimensions)):
            continue
        if coordinate is None:
            z_values = [
                z for z in possible_z if z + dimensions[2] <= state.truck.dimensions[2]
            ]
            if not z_values:
                continue
            position = (
                rng.randint(0, state.truck.dimensions[0] - dimensions[0]),
                rng.randint(0, state.truck.dimensions[1] - dimensions[1]),
                rng.choice(z_values),
            )
        else:
            position = coordinate
        placement = Placement(position, dimensions, selected_orientation)
        placements = dict(state.placements)
        placements[package.id] = placement
        if not validate_state(State(state.packages, state.truck, placements)):
            return placement
    return None
