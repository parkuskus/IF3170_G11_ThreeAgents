export type Dim3 = [number, number, number];

export interface PackageSpec {
  id: string;
  dimensions: Dim3;
  value: number;
  weight: number;
  fragile: boolean;
  eta: number;
}

export interface TruckSpec {
  dimensions: Dim3;
  max_capacity: number;
}

export interface Problem {
  trucks: TruckSpec[];
  packages: PackageSpec[];
}

export interface Example {
  name: string;
  description: string;
  problem: Problem;
}

export type ParamValue = number | string | boolean;

export interface ParamSpec {
  name: string;
  label: string;
  kind: "int" | "float" | "bool" | "select";
  default: ParamValue;
  min: number | null;
  max: number | null;
  step: number | null;
  options: string[];
  help: string;
  visibleWhen: { param: string; values: string[] } | null;
}

export interface AlgorithmSpec {
  key: string;
  label: string;
  params: ParamSpec[];
}

export interface Catalogue {
  algorithms: AlgorithmSpec[];
  objectives: { key: string; label: string }[];
}

export interface Placement {
  coordinate: Dim3;
  dimensions: Dim3;
  orientation: Dim3;
}

/** One map per truck, from package ID to where it sits (null = not in that truck). */
export type Placements = Record<string, Placement | null>[];

export interface StateResult {
  placements: Placements;
  placedWeight: number[]; // per truck
  violations: string[];
}

export interface Frame {
  evaluation: number;
  // Score of the state shown, and the best score seen up to this point.
  objective: number;
  best: number;
  placements: Placements;
}

export interface RunResult {
  algorithm: string;
  objectiveName: string;
  objectiveValue: number;
  objectiveHistory: number[];
  durationSeconds: number;
  seed: number;
  parameters: Record<string, ParamValue>;
  // Algorithm-specific; keys are the snake_case dataclass fields.
  metrics: Record<string, number | string | null | number[]>;
  initial: StateResult;
  final: StateResult;
  // Sampled search trajectory for replay; empty unless the run asked for animation.
  frames: Frame[];
  packages: PackageSpec[];
  trucks: { dimensions: Dim3; maxCapacity: number }[];
}

export interface RunRequest {
  problem: Problem;
  algorithm: string;
  objective: string;
  seed: number;
  parameters: Record<string, ParamValue>;
  animate: boolean;
}
