export interface Series {
  name: string;
  values: number[];
  color: string;
  // Draw separate points instead of a line, for noisy per-iteration values.
  dots?: boolean;
}

interface Props {
  title: string;
  xLabel: string;
  series: Series[];
  // Fixed y range instead of fitting the data.
  yDomain?: [number, number];
  // X positions marked with a dotted vertical line, e.g. where a restart begins.
  markers?: number[];
  markerLabel?: string;
}

const W = 520;
const H = 220;
const M = { top: 12, right: 14, bottom: 34, left: 56 };
const MAX_POINTS = 600;

function niceTicks(min: number, max: number, count = 4, minStep = 0): number[] {
  if (min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = Math.max(minStep, [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw)!);
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(t);
  return ticks;
}

const format = (v: number) =>
  Math.abs(v) >= 1000 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : v.toPrecision(3);

export default function LineChart({ title, xLabel, series, yDomain, markers = [], markerLabel }: Props) {
  const length = Math.max(...series.map((s) => s.values.length));
  const all = series.flatMap((s) => s.values);
  if (length === 0 || all.length === 0) return null;

  let [yMin, yMax] = yDomain ?? [Math.min(...all), Math.max(...all)];
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.05;
  yMin -= pad;
  yMax += pad;

  const x = (i: number) => M.left + (length <= 1 ? 0 : (i / (length - 1)) * (W - M.left - M.right));
  const y = (v: number) => H - M.bottom - ((v - yMin) / (yMax - yMin)) * (H - M.top - M.bottom);
  const stride = Math.max(1, Math.ceil(length / MAX_POINTS));

  const sampled = (values: number[]) => {
    const indices: number[] = [];
    for (let i = 0; i < values.length; i += stride) indices.push(i);
    // Always include the last point so the final value is not lost to downsampling.
    if ((values.length - 1) % stride !== 0) indices.push(values.length - 1);
    return indices;
  };
  // Striding would drop rare outliers, which are what a scatter is for, so keep
  // one point per distinct pixel instead.
  const dotPoints = (values: number[]) => {
    const seen = new Set<string>();
    const points: [number, number][] = [];
    values.forEach((v, i) => {
      const point: [number, number] = [Math.round(x(i)), Math.round(y(v))];
      const key = point.join();
      if (seen.has(key)) return;
      seen.add(key);
      points.push(point);
    });
    return points;
  };
  const path = (values: number[]) =>
    sampled(values)
      .map((i) => `${x(i).toFixed(1)},${y(values[i]).toFixed(1)}`)
      .join(" ");

  return (
    <figure className="chart">
      <figcaption>
        <strong>{title}</strong>
        <span className="legend">
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
          {markers.length > 0 && markerLabel && (
            <span>
              <i className="marker" />
              {markerLabel}
            </span>
          )}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {niceTicks(yMin, yMax).map((t) => (
          <g key={t}>
            <line className="grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={M.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">
              {format(t)}
            </text>
          </g>
        ))}
        {/* Iterations are whole numbers, so short runs must not get fractional ticks. */}
        {niceTicks(0, length - 1, 5, 1).map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - M.bottom + 14} textAnchor="middle">
            {Math.round(t)}
          </text>
        ))}
        <text className="tick" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          {xLabel}
        </text>
        {markers.map((m) => (
          <line key={m} className="marker" x1={x(m)} x2={x(m)} y1={M.top} y2={H - M.bottom} />
        ))}
        {series.map((s) =>
          s.values.length === 1 ? (
            <circle key={s.name} cx={x(0)} cy={y(s.values[0])} r={3.5} fill={s.color} />
          ) : s.dots ? (
            <g key={s.name} fill={s.color} fillOpacity={0.55}>
              {dotPoints(s.values).map(([cx, cy]) => (
                <circle key={`${cx},${cy}`} cx={cx} cy={cy} r={1.6} />
              ))}
            </g>
          ) : (
            <polyline key={s.name} points={path(s.values)} fill="none" stroke={s.color} strokeWidth={1.8} />
          ),
        )}
      </svg>
    </figure>
  );
}
