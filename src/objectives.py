"""Objective functions for the single-truck problem.

total_value is the required objective. The others are the bonus alternatives:
they only reuse existing problem variables (value, ETA, placement, truck size).
"""

from collections.abc import Callable

from src.model import State

# Keeps the compactness tie-breaker strictly below one value unit, so a state
# with a higher total value always scores higher.
_COMPACTNESS_SCALE = 0.99


def total_value(state: State) -> int:
    """Required objective: sum of values of packages inside the truck."""
    return state.objective()


def eta_weighted_value(state: State) -> float:
    """Value discounted by ETA: urgent packages (small ETA) count for more.

    A package with ETA e contributes value / (e + 1), so loading a package due
    today (ETA 0) earns its full value while one due in 3 days earns a quarter.
    """
    return sum(
        package.value / (package.eta + 1)
        for package in state.packages
        if state.placements[package.id] is not None
    )


def compactness(state: State) -> float:
    """Return 1 minus the volume-weighted mean top height, normalised to [0, 1].

    Packages resting low in the container score close to 1; stacks reaching the
    roof score close to 0. An empty truck scores 0.
    """
    height = state.truck.dimensions[2]
    weighted_top = 0
    total_volume = 0
    for placement in state.placements.values():
        if placement is None:
            continue
        width, length, box_height = placement.oriented_dimensions
        volume = width * length * box_height
        weighted_top += (placement.coordinate[2] + box_height) * volume
        total_volume += volume
    if total_volume == 0:
        return 0.0
    return 1 - weighted_top / (height * total_volume)


def compact_value(state: State) -> float:
    """Total value plus a sub-unit bonus for keeping packages low and compact.

    Ranks states by total value first, then breaks ties in favour of lower
    stacking. This turns the flat plateaus of total_value (moves and rotations
    that keep the same packages inside) into a gradient that leaves free space
    on top for the next package.
    """
    return state.objective() + _COMPACTNESS_SCALE * compactness(state)


OBJECTIVES: dict[str, Callable[[State], int | float]] = {
    "total_value": total_value,
    "eta_weighted_value": eta_weighted_value,
    "compact_value": compact_value,
}
