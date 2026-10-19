"""FastAPI app: JSON API for running searches, plus the built React UI."""

import json
from pathlib import Path
from random import Random

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.experiments.data import load_problem
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


@app.get("/api/examples")
def get_examples():
    """Fixture problems that load cleanly, offered as presets."""
    examples = []
    for path in sorted(FIXTURES.glob("*.json")):
        try:
            load_problem(path)
        except ValueError:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        examples.append(
            {
                "name": path.stem,
                "description": data.get("description", ""),
                "problem": {"truck": data["truck"], "packages": data["packages"]},
            }
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
        packages, truck = build_problem(request.problem)
        parameters = spec.coerce(request.parameters)
        initial = random_initial_state(packages, truck, Random(request.seed))
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
    payload["truck"] = {"dimensions": list(truck.dimensions), "maxCapacity": truck.max_capacity}
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
