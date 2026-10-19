import type { AlgorithmSpec, Catalogue, ParamSpec, ParamValue } from "../types";

interface Props {
  catalogue: Catalogue;
  algorithm: AlgorithmSpec;
  objective: string;
  seed: number;
  animate: boolean;
  values: Record<string, ParamValue>;
  running: boolean;
  onAlgorithm: (key: string) => void;
  onObjective: (key: string) => void;
  onSeed: (seed: number) => void;
  onAnimate: (animate: boolean) => void;
  onValue: (name: string, value: ParamValue) => void;
  onReset: () => void;
  onRun: () => void;
}

function isVisible(param: ParamSpec, values: Record<string, ParamValue>) {
  const rule = param.visibleWhen;
  return !rule || rule.values.includes(String(values[rule.param]));
}

function ParamField({
  param,
  value,
  onChange,
}: {
  param: ParamSpec;
  value: ParamValue;
  onChange: (value: ParamValue) => void;
}) {
  const label = <span>{param.label}</span>;
  let control;
  if (param.kind === "select") {
    control = (
      <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
        {param.options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  } else if (param.kind === "bool") {
    control = (
      <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
    );
  } else {
    control = (
      <input
        type="number"
        value={Number(value)}
        min={param.min ?? undefined}
        max={param.max ?? undefined}
        step={param.step ?? (param.kind === "int" ? 1 : "any")}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      />
    );
  }
  return (
    <label className={param.kind === "bool" ? "field inline" : "field"} title={param.help || undefined}>
      {label}
      {control}
      {param.help && <small>{param.help}</small>}
    </label>
  );
}

export default function SolverPanel(props: Props) {
  const { catalogue, algorithm, values } = props;
  return (
    <section className="card">
      <h2>Solver</h2>
      <label className="field">
        <span>Algorithm</span>
        <select value={algorithm.key} onChange={(e) => props.onAlgorithm(e.target.value)}>
          {catalogue.algorithms.map((a) => (
            <option key={a.key} value={a.key}>
              {a.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Objective function</span>
        <select value={props.objective} onChange={(e) => props.onObjective(e.target.value)}>
          {catalogue.objectives.map((o) => (
            <option key={o.key} value={o.key}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Random seed</span>
        <div className="row tight">
          <input
            type="number"
            step={1}
            value={props.seed}
            onChange={(e) => props.onSeed(Math.trunc(Number(e.target.value)))}
          />
          <button type="button" onClick={() => props.onSeed(Math.floor(Math.random() * 1_000_000))}>
            Random
          </button>
        </div>
      </label>

      <label className="field inline" title="Record the best states found during the search and replay them in the truck view">
        <input type="checkbox" checked={props.animate} onChange={(e) => props.onAnimate(e.target.checked)} />
        <span>Animate solving</span>
      </label>

      <h3>Parameters</h3>
      {algorithm.params
        .filter((p) => isVisible(p, values))
        .map((p) => (
          <ParamField key={p.name} param={p} value={values[p.name]} onChange={(v) => props.onValue(p.name, v)} />
        ))}

      <div className="row actions">
        <button type="button" className="primary" disabled={props.running} onClick={props.onRun}>
          {props.running ? "Running…" : "Run"}
        </button>
        <button type="button" onClick={props.onReset}>
          Reset parameters
        </button>
      </div>
    </section>
  );
}
