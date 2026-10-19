import { useRef } from "react";
import type { Dim3, Example, PackageSpec, Problem, TruckSpec } from "../types";

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
  if (!data || typeof data !== "object" || !Array.isArray(data.packages))
    throw new Error('JSON must contain a "packages" list');
  // A fixture has one "truck"; a fleet has a "trucks" list.
  const trucks = Array.isArray(data.trucks) ? data.trucks : data.truck ? [data.truck] : null;
  if (!trucks || trucks.length === 0) throw new Error('JSON must contain a "truck" object or a "trucks" list');
  return { trucks, packages: data.packages };
}

export default function ProblemEditor({ problem, examples, onChange, onError }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const { trucks, packages } = problem;

  const setTruck = (index: number, patch: Partial<TruckSpec>) =>
    onChange({ ...problem, trucks: trucks.map((t, i) => (i === index ? { ...t, ...patch } : t)) });
  const setTruckDim = (index: number, axis: number, value: number) => {
    const dimensions = [...trucks[index].dimensions] as Dim3;
    dimensions[axis] = value;
    setTruck(index, { dimensions });
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
      new Blob(
        [
          JSON.stringify(
            // One truck keeps the fixture shape so the file works with the CLI runners too.
            trucks.length === 1 ? { truck: trucks[0], packages } : { trucks, packages },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
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

      <div className="card-head">
        <h3>Trucks ({trucks.length})</h3>
        <button
          type="button"
          onClick={() =>
            // A new truck starts as a copy of the last one.
            onChange({ ...problem, trucks: [...trucks, structuredClone(trucks[trucks.length - 1])] })
          }
        >
          + Add truck
        </button>
      </div>
      <div className="table-wrap">
        <table className="packages">
          <thead>
            <tr>
              <th />
              <th>Width (x)</th>
              <th>Length (y)</th>
              <th>Height (z)</th>
              <th>Max weight</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trucks.map((truck, i) => (
              <tr key={i}>
                <td>Truck {i + 1}</td>
                {[0, 1, 2].map((axis) => (
                  <td key={axis}>
                    <input
                      type="number"
                      min={1}
                      aria-label={`Truck ${i + 1} ${["width", "length", "height"][axis]}`}
                      value={truck.dimensions[axis]}
                      onChange={(e) => setTruckDim(i, axis, num(e.target.value))}
                    />
                  </td>
                ))}
                <td>
                  <input
                    type="number"
                    min={0}
                    aria-label={`Truck ${i + 1} max weight`}
                    value={truck.max_capacity}
                    onChange={(e) => setTruck(i, { max_capacity: num(e.target.value) })}
                  />
                </td>
                <td>
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={`Remove truck ${i + 1}`}
                    disabled={trucks.length === 1}
                    onClick={() => onChange({ ...problem, trucks: trucks.filter((_, j) => j !== i) })}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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
