"""Objective functions for the single-truck problem.

total_value is the required objective. The others are the bonus alternatives:
they only reuse existing problem variables (value, ETA, placement, truck size).
"""

from collections.abc import Callable
from collections.abc import Mapping

from src.model import FleetState, Placement, State

# Keeps the compactness tie-breaker strictly below one value unit, so a state
# with a higher total value always scores higher.
_COMPACTNESS_SCALE = 0.99


SearchState = State | FleetState


def _placement_maps(state: SearchState) -> tuple[Mapping[str, Placement | None], ...]:
    return (state.placements,) if isinstance(state, State) else state.placements


def total_value(state: SearchState) -> int:
    """Required objective: sum of values of packages inside the truck."""
    return state.objective()


def eta_weighted_value(state: SearchState) -> float:
    """Value discounted by ETA: urgent packages (small ETA) count for more.

    A package with ETA e contributes value / (e + 1), so loading a package due
    today (ETA 0) earns its full value while one due in 3 days earns a quarter.
    """
    placements = _placement_maps(state)
    return sum(
        package.value / (package.eta + 1)
        for package in state.packages
        if any(truck_placements[package.id] is not None for truck_placements in placements)
    )


def compactness(state: SearchState) -> float:
    """Return 1 minus the volume-weighted mean top height, normalised to [0, 1].

    Packages resting low in the container score close to 1; stacks reaching the
    roof score close to 0. An empty truck scores 0.
    """
    weighted_top = 0
    height_volume = 0
    trucks = (state.truck,) if isinstance(state, State) else state.trucks
    for truck, placements in zip(trucks, _placement_maps(state)):
        truck_volume = 0
        for placement in placements.values():
            if placement is None:
                continue
            width, length, box_height = placement.oriented_dimensions
            volume = width * length * box_height
            weighted_top += (placement.coordinate[2] + box_height) * volume
            truck_volume += volume
        height_volume += truck.dimensions[2] * truck_volume
    if height_volume == 0:
        return 0.0
    return 1 - weighted_top / height_volume


def compact_value(state: SearchState) -> float:
    """Total value plus a sub-unit bonus for keeping packages low and compact.

    Ranks states by total value first, then breaks ties in favour of lower
    stacking. This turns the flat plateaus of total_value (moves and rotations
    that keep the same packages inside) into a gradient that leaves free space
    on top for the next package.
    """
    return state.objective() + _COMPACTNESS_SCALE * compactness(state)


OBJECTIVES: dict[str, Callable[[SearchState], int | float]] = {
    "total_value": total_value,
    "eta_weighted_value": eta_weighted_value,
    "compact_value": compact_value,
}
