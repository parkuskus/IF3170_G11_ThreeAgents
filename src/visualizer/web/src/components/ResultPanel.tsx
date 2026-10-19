import { useEffect, useMemo, useState } from "react";
import type { Frame, PackageSpec, Placements, RunResult, StateResult } from "../types";
import LineChart, { type Series } from "./LineChart";
import TruckView, { colorFor, type ColorMode } from "./TruckView";

const BEST = "#2f7fd1";
const SECONDARY = "#e0832b";
const TERTIARY = "#3aa17e";

const asNumbers = (value: unknown): number[] => (Array.isArray(value) ? (value as number[]) : []);

function charts(result: RunResult) {
  const m = result.metrics;
  if (m.best_objective_history !== undefined) {
    return [
      <LineChart
        key="ga"
        title="Fitness per generation"
        xLabel="generation"
        series={[
          { name: "best", values: asNumbers(m.best_objective_history), color: BEST },
          { name: "average", values: asNumbers(m.average_objective_history), color: SECONDARY },
        ]}
      />,
    ];
  }
  if (m.temperature_history !== undefined) {
    const series: Series[] = [
      { name: "best so far", values: result.objectiveHistory, color: BEST },
      { name: "current", values: asNumbers(m.current_objective_history), color: SECONDARY },
    ];
    return [
      <LineChart key="sa" title="Objective per iteration" xLabel="iteration" series={series} />,
      <LineChart
        key="temp"
        title="Temperature"
        xLabel="iteration"
        series={[{ name: "temperature", values: asNumbers(m.temperature_history), color: TERTIARY }]}
      />,
    ];
  }
  return [
    <LineChart
      key="hc"
      title="Objective per iteration"
      xLabel="iteration"
      series={[{ name: "current", values: result.objectiveHistory, color: BEST }]}
    />,
  ];
}

/** Index of the truck holding the package, or -1 when it is left out. */
const truckOf = (placements: Placements, id: string) => placements.findIndex((truck) => truck[id] != null);

const loadedCount = (state: StateResult, packages: PackageSpec[]) =>
  packages.filter((p) => truckOf(state.placements, p.id) >= 0).length;

function totalValue(state: StateResult, packages: PackageSpec[]) {
  return packages.reduce((sum, p) => sum + (truckOf(state.placements, p.id) >= 0 ? p.value : 0), 0);
}

const fmt = (value: number) => Math.round(value * 1000) / 1000;
const label = (key: string) => key.replace(/_/g, " ");

/** Consecutive frames that look identical add nothing to a replay, so drop them. */
function visibleFrames(frames: Frame[], onlyBests: boolean): Frame[] {
  const kept: Frame[] = [];
  let previous = "";
  frames.forEach((frame, index) => {
    if (onlyBests && index > 0 && frame.objective < frame.best) return;
    const signature = JSON.stringify(frame.placements);
    if (signature === previous) return;
    previous = signature;
    kept.push(frame);
  });
  return kept;
}

// One replay step takes this long at 1x: long enough to follow a package going
// in or out of the truck, and to see it turn.
const STEP_SECONDS = 1.3;

export default function ResultPanel({ result }: { result: RunResult }) {
  const { packages, trucks } = result;
  const [onlyBests, setOnlyBests] = useState(false);
  const frames = useMemo(() => visibleFrames(result.frames, onlyBests), [result.frames, onlyBests]);
  const hasReplay = result.frames.length > 1;
  const [view, setView] = useState<"initial" | "replay" | "final">(hasReplay ? "replay" : "final");
  const [colorMode, setColorMode] = useState<ColorMode>("eta");
  const [hovered, setHovered] = useState<string | null>(null);
  const [playing, setPlaying] = useState(hasReplay);
  const [frameIndex, setFrameIndex] = useState(0);
  const [speed, setSpeed] = useState(1);

  // The next step starts only once the previous one has finished moving.
  const stepSeconds = STEP_SECONDS / speed;
  useEffect(() => {
    if (view !== "replay" || !playing) return;
    const id = setInterval(
      () => setFrameIndex((i) => Math.min(i + 1, frames.length - 1)),
      stepSeconds * 1000,
    );
    return () => clearInterval(id);
  }, [view, playing, stepSeconds, frames.length]);
  useEffect(() => {
    if (playing && frameIndex >= frames.length - 1) setPlaying(false);
  }, [playing, frameIndex, frames.length]);

  const frame = frames[Math.min(frameIndex, frames.length - 1)];
  const placements =
    view === "replay" && frame ? frame.placements : result[view === "initial" ? "initial" : "final"].placements;
  const placedCount = packages.filter((p) => truckOf(placements, p.id) >= 0).length;

  const showView = (next: typeof view) => {
    setView(next);
    if (next !== "replay") setPlaying(false);
  };
  const togglePlay = () => {
    if (!playing && frameIndex >= frames.length - 1) setFrameIndex(0);
    setPlaying(!playing);
  };
  const scalarMetrics = Object.entries(result.metrics).filter(
    ([, v]) => v !== null && !Array.isArray(v),
  );

  return (
    <div className="results">
      <section className="card">
        <h2>Summary</h2>
        <dl className="stats">
          <div>
            <dt>Objective ({label(result.objectiveName)})</dt>
            <dd className="big">{Math.round(result.objectiveValue * 1000) / 1000}</dd>
          </div>
          <div>
            <dt>Total value</dt>
            <dd>
              {totalValue(result.initial, packages)} → <strong>{totalValue(result.final, packages)}</strong>
            </dd>
          </div>
          <div>
            <dt>Packages loaded</dt>
            <dd>
              {loadedCount(result.initial, packages)} → <strong>{loadedCount(result.final, packages)}</strong> /{" "}
              {packages.length}
            </dd>
          </div>
          {trucks.map((truck, i) => (
            <div key={i}>
              <dt>{trucks.length > 1 ? `Truck ${i + 1} weight` : "Weight"}</dt>
              <dd>
                {result.initial.placedWeight[i]} → <strong>{result.final.placedWeight[i]}</strong> /{" "}
                {truck.maxCapacity}
              </dd>
            </div>
          ))}
          <div>
            <dt>Duration</dt>
            <dd>{result.durationSeconds.toFixed(3)} s</dd>
          </div>
          <div>
            <dt>Seed</dt>
            <dd>{result.seed}</dd>
          </div>
          {scalarMetrics.map(([key, value]) => (
            <div key={key}>
              <dt>{label(key)}</dt>
              <dd>{String(value)}</dd>
            </div>
          ))}
        </dl>
        {(["initial", "final"] as const).map(
          (k) =>
            result[k].violations.length > 0 && (
              <p key={k} className="error" role="alert">
                {k} state violates constraints: {result[k].violations.join("; ")}
              </p>
            ),
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Truck</h2>
          <div className="row">
            <div className="segmented" role="group" aria-label="State">
              {(hasReplay ? (["initial", "replay", "final"] as const) : (["initial", "final"] as const)).map((k) => (
                <button key={k} type="button" aria-pressed={view === k} onClick={() => showView(k)}>
                  {k}
                </button>
              ))}
            </div>
            <select aria-label="Color by" value={colorMode} onChange={(e) => setColorMode(e.target.value as ColorMode)}>
              <option value="eta">Color by ETA</option>
              <option value="value">Color by value</option>
              <option value="fragile">Color by fragile</option>
            </select>
          </div>
        </div>
        <TruckView
          trucks={trucks.map((t) => t.dimensions)}
          packages={packages}
          placements={placements}
          colorMode={colorMode}
          hovered={hovered}
          stepSeconds={view === "replay" ? stepSeconds : STEP_SECONDS}
          onHover={setHovered}
        />
        {hasReplay && view === "replay" && (
          <div className="replay" role="group" aria-label="Replay controls">
            <button type="button" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
              {playing ? "❚❚" : "▶"}
            </button>
            <input
              type="range"
              min={0}
              max={frames.length - 1}
              value={frameIndex}
              aria-label="Replay position"
              onChange={(e) => {
                setPlaying(false);
                setFrameIndex(Number(e.target.value));
              }}
            />
            <select aria-label="Replay speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {[0.5, 1, 2, 4].map((v) => (
                <option key={v} value={v}>
                  {v}×
                </option>
              ))}
            </select>
            <label className="replay-option">
              <input
                type="checkbox"
                checked={onlyBests}
                onChange={(e) => {
                  setOnlyBests(e.target.checked);
                  setFrameIndex(0);
                  setPlaying(true);
                }}
              />
              New bests only
            </label>
            <span className="replay-info">
              Step {frameIndex + 1}/{frames.length} · evaluation {frame.evaluation} · {fmt(frame.objective)}
              {frame.objective >= frame.best ? " (new best)" : ` (best so far ${fmt(frame.best)})`}
            </span>
          </div>
        )}
        <p className="hint">
          Drag to rotate, scroll to zoom, hover a box to highlight it.
          {colorMode === "eta" && " Warm = due sooner."}
          {colorMode === "value" && " Warm = more valuable."}
          {colorMode === "fragile" && " Red = fragile."}
        </p>

        <div className="table-wrap">
          <table className="packages">
            <thead>
              <tr>
                <th />
                <th>ID</th>
                <th>Status</th>
                <th>Position (x,y,z)</th>
                <th>Oriented size</th>
                <th>Value</th>
                <th>Weight</th>
                <th>ETA</th>
                <th>Fragile</th>
              </tr>
            </thead>
            <tbody>
              {packages.map((p) => {
                const truckIndex = truckOf(placements, p.id);
                const placement = truckIndex >= 0 ? placements[truckIndex][p.id] : null;
                return (
                  <tr
                    key={p.id}
                    className={[hovered === p.id ? "hover" : "", placement ? "" : "muted"].join(" ")}
                    onMouseEnter={() => placement && setHovered(p.id)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <td>
                      <i className="swatch" style={{ background: colorFor(p, colorMode, packages) }} />
                    </td>
                    <td>{p.id}</td>
                    <td>{placement ? (trucks.length > 1 ? `Truck ${truckIndex + 1}` : "loaded") : "left out"}</td>
                    <td>{placement ? placement.coordinate.join(", ") : "–"}</td>
                    <td>{placement ? placement.dimensions.join(" × ") : "–"}</td>
                    <td>{p.value}</td>
                    <td>{p.weight}</td>
                    <td>{p.eta}</td>
                    <td>{p.fragile ? "yes" : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="hint">
          {placedCount} of {packages.length} packages loaded
          {view === "replay" ? " at this step" : ` in the ${view} state`}.
        </p>
      </section>

      <section className="card">
        <h2>Search progress</h2>
        <div className="charts">{charts(result)}</div>
      </section>
    </div>
  );
}
