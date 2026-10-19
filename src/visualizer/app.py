"""FastAPI app: JSON API for running searches, plus the built React UI."""

import json
from pathlib import Path
from random import Random

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.algorithms.genetic_algorithm.fleet import random_initial_fleet_state
from src.experiments.common.data import load_fleet_problem
from src.neighbors import random_initial_state
from src.objectives import OBJECTIVES
from src.visualizer.recording import FrameRecorder
from src.visualizer.registry import ALGORITHMS, catalogue
from src.visualizer.schemas import RunRequest, build_problem, result_json

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
DIST = Path(__file__).parent / "web" / "dist"

app = FastAPI(title="Package loading visualizer")


@app.get("/api/catalogue")
def get_catalogue():
    return catalogue()


# Fixtures that also get a multi-truck variant: (name, number of identical trucks).
FLEET_VARIANTS = (("base", 2), ("hard", 3))


def _example(name: str, description: str, packages, trucks) -> dict:
    return {
        "name": name,
        "description": description,
        "problem": {
            "trucks": [
                {"dimensions": list(t.dimensions), "max_capacity": t.max_capacity} for t in trucks
            ],
            "packages": [
                {
                    "id": p.id,
                    "dimensions": list(p.dimensions),
                    "value": p.value,
                    "weight": p.weight,
                    "fragile": p.fragile,
                    "eta": p.eta,
                }
                for p in packages
            ],
        },
    }


@app.get("/api/examples")
def get_examples():
    """Fixture problems that load cleanly, offered as presets."""
    examples = []
    for path in sorted(FIXTURES.glob("*.json")):
        try:
            packages, trucks = load_fleet_problem(path)
        except ValueError:
            continue
        description = json.loads(path.read_text(encoding="utf-8")).get("description", "")
        examples.append(_example(path.stem, description, packages, trucks))
        for name, count in FLEET_VARIANTS:
            if path.stem == name:
                packages, trucks = load_fleet_problem(path, truck_count=count)
                examples.append(
                    _example(f"{name}-{count}-trucks", f"{description} ({count} identical trucks)", packages, trucks)
                )
    return examples


# A plain `def` endpoint runs in FastAPI's threadpool, so a long search does not
# block the event loop.
@app.post("/api/run")
def run(request: RunRequest):
    spec = ALGORITHMS.get(request.algorithm)
    if spec is None:
        raise HTTPException(422, f"unknown algorithm {request.algorithm!r}")
    objective = OBJECTIVES.get(request.objective)
    if objective is None:
        raise HTTPException(422, f"unknown objective {request.objective!r}")
    try:
        packages, trucks = build_problem(request.problem)
        parameters = spec.coerce(request.parameters)
        rng = Random(request.seed)
        initial = (
            random_initial_state(packages, trucks[0], rng)
            if len(trucks) == 1
            else random_initial_fleet_state(packages, trucks, rng)
        )
        recorder = FrameRecorder(objective, initial) if request.animate else None
        result = spec.run(
            initial,
            seed=request.seed,
            objective_fn=recorder or objective,
            objective_name=request.objective,
            **parameters,
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    payload = result_json(result)
    payload["frames"] = (
        recorder.frames(result.final_state, result.objective_value) if recorder else []
    )
    payload["packages"] = [
        {
            "id": p.id,
            "dimensions": list(p.dimensions),
            "value": p.value,
            "weight": p.weight,
            "fragile": p.fragile,
            "eta": p.eta,
        }
        for p in packages
    ]
    payload["trucks"] = [
        {"dimensions": list(t.dimensions), "maxCapacity": t.max_capacity} for t in trucks
    ]
    return payload


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = (DIST / path).resolve()
        if path and candidate.is_file() and DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
