import { useRef } from "react";
import type { Dim3, Example, PackageSpec, Problem } from "../types";

interface Props {
  problem: Problem;
  examples: Example[];
  onChange: (problem: Problem) => void;
  onError: (message: string) => void;
}

const num = (value: string) => (value === "" ? 0 : Number(value));

function nextId(packages: PackageSpec[]): string {
  let n = packages.length + 1;
  while (packages.some((p) => p.id === `Box-${n}`)) n++;
  return `Box-${n}`;
}

function parseProblem(text: string): Problem {
  const data = JSON.parse(text);
  if (!data || typeof data !== "object" || !data.truck || !Array.isArray(data.packages))
    throw new Error('JSON must contain a "truck" object and a "packages" list');
  return { truck: data.truck, packages: data.packages };
}

export default function ProblemEditor({ problem, examples, onChange, onError }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const { truck, packages } = problem;

  const setTruckDim = (axis: number, value: number) => {
    const dimensions = [...truck.dimensions] as Dim3;
    dimensions[axis] = value;
    onChange({ ...problem, truck: { ...truck, dimensions } });
  };
  const setPackage = (index: number, patch: Partial<PackageSpec>) =>
    onChange({
      ...problem,
      packages: packages.map((p, i) => (i === index ? { ...p, ...patch } : p)),
    });
  const setPackageDim = (index: number, axis: number, value: number) => {
    const dimensions = [...packages[index].dimensions] as Dim3;
    dimensions[axis] = value;
    setPackage(index, { dimensions });
  };

  const importFile = async (file: File) => {
    try {
      onChange(parseProblem(await file.text()));
    } catch (error) {
      onError(`Could not import ${file.name}: ${(error as Error).message}`);
    }
  };

  const exportFile = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(problem, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "problem.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="card">
      <div className="card-head">
        <h2>Problem</h2>
        <div className="row">
          <select
            aria-label="Load example"
            value=""
            onChange={(e) => {
              const example = examples.find((x) => x.name === e.target.value);
              if (example) onChange(structuredClone(example.problem));
            }}
          >
            <option value="">Load example…</option>
            {examples.map((e) => (
              <option key={e.name} value={e.name} title={e.description}>
                {e.name}
              </option>
            ))}
          </select>
          <button type="button" onClick={() => fileInput.current?.click()}>
            Import JSON
          </button>
          <button type="button" onClick={exportFile}>
            Export JSON
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void importFile(file);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <h3>Truck</h3>
      <div className="row truck-row">
        {(["Width (x)", "Length (y)", "Height (z)"] as const).map((label, axis) => (
          <label key={label} className="field">
            <span>{label}</span>
            <input
              type="number"
              min={1}
              value={truck.dimensions[axis]}
              onChange={(e) => setTruckDim(axis, num(e.target.value))}
            />
          </label>
        ))}
        <label className="field">
          <span>Max weight</span>
          <input
            type="number"
            min={0}
            value={truck.max_capacity}
            onChange={(e) =>
              onChange({ ...problem, truck: { ...truck, max_capacity: num(e.target.value) } })
            }
          />
        </label>
      </div>

      <div className="card-head">
        <h3>Packages ({packages.length})</h3>
        <button
          type="button"
          onClick={() =>
            onChange({
              ...problem,
              packages: [
                ...packages,
                { id: nextId(packages), dimensions: [1, 1, 1], value: 10, weight: 1, fragile: false, eta: 1 },
              ],
            })
          }
        >
          + Add package
        </button>
      </div>
      <div className="table-wrap">
        <table className="packages">
          <thead>
            <tr>
              <th>ID</th>
              <th>W</th>
              <th>L</th>
              <th>H</th>
              <th>Value</th>
              <th>Weight</th>
              <th>ETA</th>
              <th>Fragile</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {packages.map((p, i) => (
              <tr key={i}>
                <td>
                  <input
                    className="id-input"
                    aria-label="Package ID"
                    value={p.id}
                    onChange={(e) => setPackage(i, { id: e.target.value })}
                  />
                </td>
                {[0, 1, 2].map((axis) => (
                  <td key={axis}>
                    <input
                      type="number"
                      min={1}
                      aria-label={`Dimension ${"WLH"[axis]}`}
                      value={p.dimensions[axis]}
                      onChange={(e) => setPackageDim(i, axis, num(e.target.value))}
                    />
                  </td>
                ))}
                <td>
                  <input
                    type="number"
                    min={0}
                    aria-label="Value"
                    value={p.value}
                    onChange={(e) => setPackage(i, { value: num(e.target.value) })}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min={0}
                    aria-label="Weight"
                    value={p.weight}
                    onChange={(e) => setPackage(i, { weight: num(e.target.value) })}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min={0}
                    aria-label="ETA"
                    value={p.eta}
                    onChange={(e) => setPackage(i, { eta: num(e.target.value) })}
                  />
                </td>
                <td className="center">
                  <input
                    type="checkbox"
                    aria-label="Fragile"
                    checked={p.fragile}
                    onChange={(e) => setPackage(i, { fragile: e.target.checked })}
                  />
                </td>
                <td>
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={`Remove ${p.id}`}
                    onClick={() => onChange({ ...problem, packages: packages.filter((_, j) => j !== i) })}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
