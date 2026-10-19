export interface Series {
  name: string;
  values: number[];
  color: string;
}

interface Props {
  title: string;
  xLabel: string;
  series: Series[];
}

const W = 520;
const H = 220;
const M = { top: 12, right: 14, bottom: 34, left: 56 };
const MAX_POINTS = 600;

function niceTicks(min: number, max: number, count = 4): number[] {
  if (min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw)!;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(t);
  return ticks;
}

const format = (v: number) =>
  Math.abs(v) >= 1000 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : v.toPrecision(3);

export default function LineChart({ title, xLabel, series }: Props) {
  const length = Math.max(...series.map((s) => s.values.length));
  const all = series.flatMap((s) => s.values);
  if (length === 0 || all.length === 0) return null;

  let yMin = Math.min(...all);
  let yMax = Math.max(...all);
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

  const path = (values: number[]) => {
    const points: string[] = [];
    for (let i = 0; i < values.length; i += stride) points.push(`${x(i).toFixed(1)},${y(values[i]).toFixed(1)}`);
    // Always include the last point so the final value is not lost to downsampling.
    if ((values.length - 1) % stride !== 0)
      points.push(`${x(values.length - 1).toFixed(1)},${y(values[values.length - 1]).toFixed(1)}`);
    return points.join(" ");
  };

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
        {niceTicks(0, length - 1, 5).map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - M.bottom + 14} textAnchor="middle">
            {Math.round(t)}
          </text>
        ))}
        <text className="tick" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          {xLabel}
        </text>
        {series.map((s) =>
          s.values.length === 1 ? (
            <circle key={s.name} cx={x(0)} cy={y(s.values[0])} r={3.5} fill={s.color} />
          ) : (
            <polyline key={s.name} points={path(s.values)} fill="none" stroke={s.color} strokeWidth={1.8} />
          ),
        )}
      </svg>
    </figure>
  );
}
