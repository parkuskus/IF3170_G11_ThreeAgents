"""Random valid initial states and neighbors for the single-truck search."""

from random import Random

from src.model import ORIENTATIONS, Package, Placement, State, Truck
from src.validation import validate_state


def _make_state(
    packages: tuple[Package, ...], truck: Truck, placements: dict[str, Placement | None]
) -> State:
    return State(packages, truck, placements)


def _valid(state: State) -> bool:
    return not validate_state(state)


def _candidate_position(      
    package: Package,
    orientation: tuple[int, int, int],
    truck: Truck,
    placements: dict[str, Placement | None],
    rng: Random,
) -> Placement:
    dimensions = package.oriented_dimensions(orientation)
    possible_z = {0}
    for existing in placements.values():
        if existing is not None:
            possible_z.add(existing.coordinate[2] + existing.oriented_dimensions[2])
    possible_z = {z for z in possible_z if z + dimensions[2] <= truck.dimensions[2]}
    z = rng.choice(tuple(possible_z))
    x = rng.randint(0, truck.dimensions[0] - dimensions[0])
    y = rng.randint(0, truck.dimensions[1] - dimensions[1])
    return Placement((x, y, z), dimensions, orientation)


def random_initial_state(
    packages: tuple[Package, ...], truck: Truck, rng: Random | None = None, attempts_per_package: int = 50
) -> State:
    """Build a random valid state, leaving packages outside if placement fails."""
    rng = rng or Random()
    placements: dict[str, Placement | None] = {package.id: None for package in packages}
    order = list(packages)
    rng.shuffle(order)

    for package in order:
        for _ in range(attempts_per_package):
            orientation = rng.choice(ORIENTATIONS)
            dimensions = package.oriented_dimensions(orientation)
            if any(size > limit for size, limit in zip(dimensions, truck.dimensions)):
                continue
            candidate_placements = placements.copy()
            candidate_placements[package.id] = _candidate_position(
                package, orientation, truck, placements, rng
            )
            candidate = _make_state(packages, truck, candidate_placements)
            if _valid(candidate):
                placements = candidate_placements
                break

    return _make_state(packages, truck, placements)


def _swap_candidate(state: State, rng: Random) -> State:
    a, b = rng.sample(state.packages, 2)
    placement_a, placement_b = state.placements[a.id], state.placements[b.id]
    placements = dict(state.placements)
    placements[a.id] = (
        None
        if placement_b is None
        else Placement(
            placement_b.coordinate,
            a.oriented_dimensions(placement_a.orientation if placement_a else (0, 1, 2)),
            placement_a.orientation if placement_a else (0, 1, 2),
        )
    )
    placements[b.id] = (
        None
        if placement_a is None
        else Placement(
            placement_a.coordinate,
            b.oriented_dimensions(placement_b.orientation if placement_b else (0, 1, 2)),
            placement_b.orientation if placement_b else (0, 1, 2),
        )
    )
    return _make_state(state.packages, state.truck, placements)


def _move_candidate(state: State, rng: Random) -> State:
    package = rng.choice(state.packages)
    old = state.placements[package.id]
    orientation = old.orientation if old else rng.choice(ORIENTATIONS)
    dimensions = package.oriented_dimensions(orientation)
    if any(size > limit for size, limit in zip(dimensions, state.truck.dimensions)):
        return state
    placements = dict(state.placements)
    placements[package.id] = _candidate_position(
        package, orientation, state.truck, {key: value for key, value in placements.items() if key != package.id}, rng
    )
    return _make_state(state.packages, state.truck, placements)


def _rotate_candidate(state: State, rng: Random) -> State:
    placed = [package for package in state.packages if state.placements[package.id] is not None]
    if not placed:
        return state
    package = rng.choice(placed)
    old = state.placements[package.id]
    assert old is not None
    options = [orientation for orientation in ORIENTATIONS if orientation != old.orientation]
    orientation = rng.choice(options)
    placements = dict(state.placements)
    placements[package.id] = Placement(
        old.coordinate, package.oriented_dimensions(orientation), orientation
    )
    return _make_state(state.packages, state.truck, placements)


def random_neighbor(state: State, rng: Random | None = None, max_attempts: int = 100) -> State:
    """Return a valid neighbor using swap, move, or 90-degree rotation.

    If no valid candidate is found within max_attempts, return the input state.
    """
    if max_attempts < 1 or not state.packages:
        return state
    rng = rng or Random()
    operations = [_move_candidate, _rotate_candidate]
    if len(state.packages) > 1:
        operations.append(_swap_candidate)

    for _ in range(max_attempts):
        candidate = rng.choice(operations)(state, rng)
        if candidate is not state and _valid(candidate):
            return candidate
    return state
