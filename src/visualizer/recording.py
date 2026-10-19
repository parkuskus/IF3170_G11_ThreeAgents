"""Capture the search trajectory for replay without touching the algorithms.

Every algorithm scores each candidate through the `objective_fn` it is given, so
wrapping that function sees every state the search evaluates. Two kinds of
frames are kept: each state that beats the best score so far (the "climb"), and
an evenly spaced sample of all other candidates (the "exploring").
"""

from collections.abc import Callable
from typing import Any

from src.model import State
from src.visualizer.schemas import placements_json

# Few enough frames that a replay can show every step slowly enough to follow.
MAX_SAMPLES = 40
MAX_IMPROVEMENTS = 40


def _thin(frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Evenly keep `limit` frames, always including the first and the last."""
    if len(frames) <= limit:
        return frames
    last = len(frames) - 1
    keep = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [frames[i] for i in keep]


class FrameRecorder:
    def __init__(self, objective: Callable[[State], int | float], initial: State) -> None:
        self._objective = objective
        self._evaluations = 0
        self._best = objective(initial)
        self._improvements = [self._frame(initial, self._best)]
        self._samples: list[dict[str, Any]] = []
        # Keep every `_stride`-th evaluation; the stride doubles whenever the
        # sample list fills up, so memory stays bounded for any run length.
        self._stride = 1

    def _frame(self, state: State, score: int | float) -> dict[str, Any]:
        return {
            "evaluation": self._evaluations,
            "objective": score,
            "best": self._best,
            "placements": placements_json(state),
        }

    def __call__(self, state: State) -> int | float:
        score = self._objective(state)
        self._evaluations += 1
        frame = None
        if score > self._best:
            self._best = score
            frame = self._frame(state, score)
            self._improvements.append(frame)
        if self._evaluations % self._stride == 0:
            self._samples.append(frame or self._frame(state, score))
            if len(self._samples) >= 2 * MAX_SAMPLES:
                self._samples = self._samples[1::2]
                self._stride *= 2
        return score

    def frames(self, final: State, final_score: int | float) -> list[dict[str, Any]]:
        merged = {frame["evaluation"]: frame for frame in self._samples}
        for frame in _thin(self._improvements, MAX_IMPROVEMENTS):
            merged[frame["evaluation"]] = frame
        frames = [merged[evaluation] for evaluation in sorted(merged)]
        # The replay always ends on the state the algorithm returned.
        self._best = max(self._best, final_score)
        frames.append(self._frame(final, final_score))
        return frames
