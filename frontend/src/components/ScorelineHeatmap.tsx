import { useState } from "react";

interface ScorelineEntry {
  home: number;
  away: number;
  prob: number;
}

interface Props {
  grid: number[][];
  homeTeam: string;
  awayTeam: string;
  topScorelines: ScorelineEntry[];
}

type Region = "all" | "home" | "draw" | "away";

const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

/** The answer first (the five most likely scores), the grid as the evidence.
 *  Hover or tap a cell for its chance; the region switch shades the cells that
 *  make a home win, a draw or an away win and prints their share, which is read
 *  off the same grid so the three always add to 100%. */
export function ScorelineHeatmap({ grid, homeTeam, awayTeam, topScorelines }: Props) {
  const n = grid.length;
  const [region, setRegion] = useState<Region>("all");
  const [picked, setPicked] = useState<{ h: number; a: number } | null>(null);

  let peakH = 0;
  let peakA = 0;
  let max = 0;
  let total = 0;
  const share = { home: 0, draw: 0, away: 0 };
  for (let h = 0; h < n; h++) {
    for (let a = 0; a < n; a++) {
      const v = grid[h][a];
      total += v;
      if (h > a) share.home += v;
      else if (h === a) share.draw += v;
      else share.away += v;
      if (v > max) {
        max = v;
        peakH = h;
        peakA = a;
      }
    }
  }
  const norm = (v: number) => (total > 0 ? v / total : 0);
  const inRegion = (h: number, a: number) =>
    region === "all" || (region === "home" ? h > a : region === "draw" ? h === a : h < a);

  const cellColor = (v: number) => {
    const t = max > 0 ? v / max : 0;
    return `color-mix(in srgb, var(--color-pr-accent) ${Math.round(8 + t * 82)}%, var(--color-pr-panel-2))`;
  };

  const top = (topScorelines.length ? topScorelines : [{ home: peakH, away: peakA, prob: max }]).slice(0, 5);
  const readout = picked
    ? `${homeTeam} ${picked.h}–${picked.a} ${awayTeam}: ${pct1(grid[picked.h][picked.a])} chance`
    : `Most likely: ${homeTeam} ${peakH}–${peakA} ${awayTeam}, ${pct1(max)} chance`;

  const regions: { id: Region; label: string; value?: number }[] = [
    { id: "all", label: "All scores" },
    { id: "home", label: `${homeTeam} win`, value: norm(share.home) },
    { id: "draw", label: "Draw", value: norm(share.draw) },
    { id: "away", label: `${awayTeam} win`, value: norm(share.away) },
  ];

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-pr-text-faint">Most likely scores</p>
      <ol className="mb-5 flex flex-col gap-1">
        {top.map((s, i) => (
          <li key={i} className="flex items-center gap-3 rounded-pr bg-pr-panel px-3 py-2 text-sm">
            <span className="w-4 text-xs tabular-nums text-pr-text-faint">{i + 1}</span>
            <span className={`font-display text-lg tracking-wide ${i === 0 ? "text-pr-text" : "text-pr-text-dim"}`}>
              {s.home}–{s.away}
            </span>
            <span className="h-1.5 flex-1 overflow-hidden rounded-pr bg-pr-rule/60">
              <span
                className="block h-full"
                style={{ width: `${top[0].prob > 0 ? (s.prob / top[0].prob) * 100 : 0}%`, background: i === 0 ? "var(--color-pr-accent)" : "var(--color-pr-fill-mute)" }}
              />
            </span>
            <span className="w-14 text-right font-semibold tabular-nums text-pr-text">{pct1(s.prob)}</span>
          </li>
        ))}
      </ol>

      <div className="mb-2 flex flex-wrap items-center gap-1.5" role="group" aria-label="Shade the grid by result">
        {regions.map((r) => (
          <button
            key={r.id}
            type="button"
            aria-pressed={region === r.id}
            onClick={() => setRegion(r.id)}
            className={`rounded-pr border px-2.5 py-1 text-xs font-semibold transition ${
              region === r.id ? "border-pr-accent bg-pr-accent text-pr-accent-ink" : "border-pr-rule text-pr-text-dim hover:text-pr-text"
            }`}
          >
            {r.label}
            {r.value !== undefined && <span className="ml-1.5 tabular-nums">{Math.round(r.value * 100)}%</span>}
          </button>
        ))}
      </div>
      <p className="mb-1 text-xs text-pr-text-faint">
        Shares come from the scoreline model, so they can differ by a few points from the match-result bar above.
      </p>
      <p aria-live="polite" className="mb-2 min-h-5 text-sm text-pr-text-dim">
        {readout}
      </p>

      <div className="mx-auto max-w-[460px]">
      <div className="mb-1 pl-6 text-center text-xs font-medium text-pr-text-faint">{awayTeam} goals →</div>
      <div className="flex gap-1">
        <div className="flex flex-col items-center justify-center pr-0.5">
          <span className="-rotate-90 whitespace-nowrap text-xs font-medium text-pr-text-faint">{homeTeam} goals ↓</span>
        </div>
        <div className="flex flex-1 flex-col gap-1">
          {grid.map((row, h) => (
            <div key={h} className="flex items-center gap-1">
              <span className="w-3 shrink-0 text-center text-xs text-pr-text-faint">{h}</span>
              <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
                {row.map((v, a) => {
                  const isPeak = h === peakH && a === peakA;
                  const active = inRegion(h, a);
                  const ink = active && max > 0 && v / max > 0.55 ? "text-pr-accent-ink" : "text-pr-text";
                  return (
                    <button
                      type="button"
                      key={a}
                      onMouseEnter={() => setPicked({ h, a })}
                      onFocus={() => setPicked({ h, a })}
                      onClick={() => setPicked({ h, a })}
                      aria-label={`${homeTeam} ${h}, ${awayTeam} ${a}: ${pct1(v)}`}
                      className={`flex aspect-square flex-col items-center justify-center rounded-pr text-xs font-semibold tabular-nums transition-opacity ${ink} ${
                        isPeak ? "ring-2 ring-pr-text/80" : ""
                      } ${picked && picked.h === h && picked.a === a ? "outline outline-2 outline-pr-text" : ""}`}
                      style={{ background: cellColor(v), opacity: active ? 1 : 0.28 }}
                    >
                      {(v * 100).toFixed(1)}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          <div className="flex gap-1 pl-4">
            {grid[0].map((_, a) => (
              <span key={a} className="flex-1 text-center text-xs text-pr-text-faint">
                {a}
              </span>
            ))}
          </div>
        </div>
      </div>
      </div>
      <p className="mt-2 text-xs text-pr-text-faint">
        Each cell is one possible final score: read down for {homeTeam}&apos;s goals, across for {awayTeam}&apos;s. Brighter is
        more likely. The ringed cell is the single most probable score.
      </p>
    </div>
  );
}
