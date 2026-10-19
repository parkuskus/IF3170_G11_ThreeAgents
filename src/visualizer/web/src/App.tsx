import { useEffect, useState } from "react";
import { getCatalogue, getExamples, runSearch } from "./api";
import ProblemEditor from "./components/ProblemEditor";
import ResultPanel from "./components/ResultPanel";
import SolverPanel from "./components/SolverPanel";
import type { AlgorithmSpec, Catalogue, Example, ParamValue, Problem, RunResult } from "./types";

const EMPTY_PROBLEM: Problem = {
  truck: { dimensions: [10, 10, 6], max_capacity: 20 },
  packages: [],
};

const defaultsOf = (algorithm: AlgorithmSpec) =>
  Object.fromEntries(algorithm.params.map((p) => [p.name, p.default])) as Record<string, ParamValue>;

export default function App() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [problem, setProblem] = useState<Problem>(EMPTY_PROBLEM);
  const [algorithmKey, setAlgorithmKey] = useState("");
  const [objective, setObjective] = useState("total_value");
  const [seed, setSeed] = useState(0);
  const [animate, setAnimate] = useState(true);
  // Bumped per run so the result view restarts its replay.
  const [runId, setRunId] = useState(0);
  // Parameter values are kept per algorithm so switching back does not lose edits.
  const [params, setParams] = useState<Record<string, Record<string, ParamValue>>>({});
  const [result, setResult] = useState<RunResult | null>(null);
  const [tab, setTab] = useState<"problem" | "result">("problem");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getCatalogue(), getExamples()])
      .then(([cat, ex]) => {
        setCatalogue(cat);
        setExamples(ex);
        setAlgorithmKey(cat.algorithms[0].key);
        setParams(Object.fromEntries(cat.algorithms.map((a) => [a.key, defaultsOf(a)])));
        const start = ex.find((e) => e.name === "demo") ?? ex[0];
        if (start) setProblem(structuredClone(start.problem));
      })
      .catch((e: Error) => setError(`Cannot reach the server: ${e.message}`));
  }, []);

  if (!catalogue || !algorithmKey) {
    return (
      <main className="app">
        <h1>Truck Loading Visualizer</h1>
        {error ? <p className="error" role="alert">{error}</p> : <p>Loading…</p>}
      </main>
    );
  }

  const algorithm = catalogue.algorithms.find((a) => a.key === algorithmKey)!;

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(
        await runSearch({ problem, algorithm: algorithmKey, objective, seed, animate, parameters: params[algorithmKey] }),
      );
      setRunId((id) => id + 1);
      setTab("result");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="app">
      <header>
        <h1>Truck Loading Visualizer</h1>
        <p>Define a package set, pick a search algorithm, and inspect how it fills the truck.</p>
      </header>
      {error && (
        <p className="error banner" role="alert">
          {error}
        </p>
      )}
      <div className="layout">
        <aside>
          <SolverPanel
            catalogue={catalogue}
            algorithm={algorithm}
            objective={objective}
            seed={seed}
            animate={animate}
            values={params[algorithmKey]}
            running={running}
            onAlgorithm={setAlgorithmKey}
            onObjective={setObjective}
            onSeed={setSeed}
            onAnimate={setAnimate}
            onValue={(name, value) =>
              setParams((prev) => ({ ...prev, [algorithmKey]: { ...prev[algorithmKey], [name]: value } }))
            }
            onReset={() => setParams((prev) => ({ ...prev, [algorithmKey]: defaultsOf(algorithm) }))}
            onRun={run}
          />
        </aside>
        <div className="main">
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "problem"} onClick={() => setTab("problem")}>
              Problem
            </button>
            <button role="tab" aria-selected={tab === "result"} disabled={!result} onClick={() => setTab("result")}>
              Result
            </button>
          </div>
          {tab === "problem" || !result ? (
            <ProblemEditor problem={problem} examples={examples} onChange={setProblem} onError={setError} />
          ) : (
            <ResultPanel key={runId} result={result} />
          )}
        </div>
      </div>
    </main>
  );
}
