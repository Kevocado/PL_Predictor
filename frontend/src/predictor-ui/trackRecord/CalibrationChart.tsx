// Synced from predictor-ui@130ae20c8e7e. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState } from "react";
import type { CalibrationBucket, CalibrationSeries } from "./types";

/** "50-60%" -> 0.55, "70%+" -> 0.8: where the bucket sits on the stated-confidence axis. */
function bucketMid(label: string): number | null {
  const m = /(\d+)\s*-\s*(\d+)/.exec(label);
  if (m) return (Number(m[1]) + Number(m[2])) / 200;
  const plus = /(\d+)\s*%?\s*\+/.exec(label);
  return plus ? (Number(plus[1]) + 10) / 100 : null;
}

const THIN = 5;

/**
 * Stated confidence against what actually happened. On the diagonal =
 * trustworthy. Ported from NBA, with `n`/`correct` in place of the API's
 * `total_predictions`/`correct_predictions`.
 */
export function CalibrationChart({ series }: { series: CalibrationSeries[] }) {
  const usable = series.filter((s) => s.buckets.some((b) => b.n > 0));
  const [pick, setPick] = useState(usable[0]?.key);
  if (usable.length === 0) return <p className="text-sm text-pr-text-dim">No confidence buckets graded yet.</p>;
  const active = usable.find((s) => s.key === pick) ?? usable[0];
  const W = 320;
  const H = 220;
  const L = 34;
  const B = 26;
  const T = 8;
  const R = 8;
  const x = (c: number) => L + ((c - 0.45) / 0.45) * (W - L - R);
  const y = (v: number) => H - B - v * (H - B - T);
  const pts = active.buckets
    .map((b) => ({ b, mid: bucketMid(b.bucket), rate: b.n > 0 ? b.correct / b.n : null }))
    .filter((p): p is { b: CalibrationBucket; mid: number; rate: number } => p.mid != null && p.b.n > 0 && p.rate != null);
  return (
    <div>
      {usable.length > 1 && (
        <div role="group" aria-label="Market" className="mb-3 inline-flex rounded-pr border border-pr-rule p-0.5">
          {usable.map((s) => (
            <button
              key={s.key}
              type="button"
              aria-pressed={s.key === active.key}
              onClick={() => setPick(s.key)}
              className={`rounded-pr px-3 py-1 text-xs font-semibold ${s.key === active.key ? "bg-pr-panel-2 text-pr-text" : "text-pr-text-dim hover:text-pr-text"}`}
            >
              {s.label}
            </button>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md" role="img" aria-label={`Calibration for ${active.label}`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="stroke-pr-rule" strokeWidth={1} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" className="fill-pr-text-faint text-[10px]">{Math.round(v * 100)}%</text>
          </g>
        ))}
        {/* The diagonal: stated confidence = actual hit rate. */}
        <line x1={x(0.45)} y1={y(0.45)} x2={x(0.9)} y2={y(0.9)} className="stroke-pr-text-faint" strokeDasharray="4 4" strokeWidth={1} />
        {[0.5, 0.6, 0.7, 0.8].map((c) => (
          <text key={c} x={x(c)} y={H - 8} textAnchor="middle" className="fill-pr-text-faint text-[10px]">{Math.round(c * 100)}%</text>
        ))}
        {pts.map(({ b, mid, rate }) => {
          const thin = b.n < THIN;
          return (
            <circle
              key={b.bucket}
              data-testid="calibration-point"
              cx={x(mid)}
              cy={y(rate)}
              r={5 + Math.min(10, b.n)}
              className={thin ? "fill-pr-accent/25 stroke-pr-accent" : "fill-pr-accent/60 stroke-pr-accent"}
              strokeWidth={1.5}
            >
              <title>{`${b.bucket} confidence: ${b.correct} of ${b.n} right`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="mt-1 text-xs text-pr-text-faint">
        Across: how sure the model said it was. Up: how often it was right. On the dashed line is honest; bigger dots
        rest on more picks; faint dots rest on fewer than {THIN}.
      </p>
    </div>
  );
}
