# Visualizer

Web interface to define a package-loading problem, pick an algorithm and its settings, and inspect the result (3D truck view, summary, progress charts).

## Run

```bash
python -m pip install -r requirements-visualizer.txt
cd src/visualizer/web && npm install && npm run build && cd ../../..
python -m src.visualizer            # http://127.0.0.1:8000
```

`web/dist/` is the built UI and is served by the Python app; it is git-ignored, so rebuild after pulling.

## Develop

Run `python -m src.visualizer --reload` and, in `src/visualizer/web`, `npm run dev` (http://localhost:5173, `/api` is proxied to port 8000).

## Multiple trucks

The Problem tab holds a list of trucks (add, remove, edit each one's size and weight limit). With one truck the search runs on a single-truck `State`; with more it runs on a `FleetState` using the fleet operators, so Hill-climbing, Simulated Annealing and the Genetic Algorithm all work on a fleet. The 3D view lays the trucks side by side, and a package that changes truck is lifted out of one and lowered into the other. JSON import/export accepts either the fixture shape (`"truck": {...}`) or `"trucks": [...]`. The `base-2-trucks` and `hard-3-trucks` examples repeat the fixture's truck.

## Solving animation

The **Animate solving** checkbox in the Solver panel (on by default) makes the server record a replay of the search; when off, no frames are recorded and the result view is static. `recording.py` does this by wrapping the `objective_fn` passed to the algorithm, so the algorithms themselves are untouched. A replay holds each new best state plus an even sample of the other candidates the search evaluated (about 300 frames at most, whatever the run length). The result view plays it with play/pause, a scrubber, and a speed selector.

## Layout

- `app.py` – FastAPI app: `GET /api/catalogue`, `GET /api/examples` (fixtures in `tests/fixtures`), `POST /api/run`.
- `registry.py` – each algorithm's callable and parameter spec. The UI builds its forms from this, so a new algorithm or parameter only needs an entry here.
- `schemas.py` – request validation and result serialisation.
- `web/` – Vite + React + TypeScript + three.js frontend.
