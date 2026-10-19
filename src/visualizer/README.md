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

## Solving animation

The **Animate solving** checkbox in the Solver panel (on by default) makes the server record a replay of the search; when off, no frames are recorded and the result view is static. `recording.py` does this by wrapping the `objective_fn` passed to the algorithm, so the algorithms themselves are untouched. A replay holds each new best state plus an even sample of the other candidates the search evaluated (about 300 frames at most, whatever the run length). The result view plays it with play/pause, a scrubber, and a speed selector.

## Layout

- `app.py` – FastAPI app: `GET /api/catalogue`, `GET /api/examples` (fixtures in `tests/fixtures`), `POST /api/run`.
- `registry.py` – each algorithm's callable and parameter spec. The UI builds its forms from this, so a new algorithm or parameter only needs an entry here.
- `schemas.py` – request validation and result serialisation.
- `web/` – Vite + React + TypeScript + three.js frontend.
