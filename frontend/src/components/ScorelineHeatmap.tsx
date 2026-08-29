import { MarketBar } from "./MarketBar";

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

export function ScorelineHeatmap({ grid, homeTeam, awayTeam, topScorelines }: Props) {
  const n = grid.length;

  let peakH = 0;
  let peakA = 0;
  let max = 0;
  for (let h = 0; h < n; h++) {
    for (let a = 0; a < n; a++) {
      if (grid[h][a] > max) {
        max = grid[h][a];
        peakH = h;
        peakA = a;
      }
    }
  }

  const cellColor = (v: number) => {
    const t = max > 0 ? v / max : 0;
    // interpolate pl-850 -> pl-pink across the intensity range
    const r = Math.round(35 + t * (233 - 35));
    const g = Math.round(15 + t * (0 - 15));
    const b = Math.round(61 + t * (82 - 61));
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div>
      <div className="mb-3 flex items-baseline gap-1.5 rounded-lg bg-pl-pink/10 px-3 py-2 ring-1 ring-pl-pink/30">
        <span className="text-xs text-pl-text-dim">Most likely scoreline:</span>
        <span className="font-mono text-base font-bold text-pl-text">
          {homeTeam} {peakH}–{peakA} {awayTeam}
        </span>
        <span className="text-xs text-pl-cyan">{(max * 100).toFixed(1)}%</span>
      </div>

      <div className="mb-1 pl-6 text-center text-[11px] font-medium text-pl-text-faint">{awayTeam} goals →</div>
      <div className="flex gap-1">
        <div className="flex flex-col items-center justify-center pr-0.5">
          <span className="-rotate-90 whitespace-nowrap text-[11px] font-medium text-pl-text-faint">{homeTeam} goals ↓</span>
        </div>
        <div className="flex flex-1 flex-col gap-1">
          {grid.map((row, h) => (
            <div key={h} className="flex items-center gap-1">
              <span className="w-3 shrink-0 text-center text-[10px] text-pl-text-faint">{h}</span>
              <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
                {row.map((v, a) => {
                  const isPeak = h === peakH && a === peakA;
                  return (
                    <div
                      key={a}
                      className={`flex aspect-square flex-col items-center justify-center rounded-md text-[11px] font-semibold text-white ${
                        isPeak ? "ring-2 ring-white/80" : ""
                      }`}
                      style={{ background: cellColor(v) }}
                      title={`${homeTeam} ${h}-${a} ${awayTeam}: ${(v * 100).toFixed(1)}%`}
                    >
                      {(v * 100).toFixed(1)}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
          <div className="flex gap-1 pl-4">
            {grid[0].map((_, a) => (
              <span key={a} className="flex-1 text-center text-[10px] text-pl-text-faint">
                {a}
              </span>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-2 text-[11px] text-pl-text-faint">
        Each cell is one possible final score — read down for {homeTeam}'s goals, across for {awayTeam}'s. Brighter =
        more likely; the highlighted cell is the single most probable score.
      </p>

      {topScorelines.length > 1 && (
        <div className="mt-4">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
            Next most likely scorelines
          </p>
          <div className="flex flex-col gap-1">
            {topScorelines.map((s, i) => (
              <MarketBar
                key={i}
                label={
                  <span className="font-mono font-medium text-pl-text">
                    {homeTeam} {s.home}–{s.away} {awayTeam}
                  </span>
                }
                prob={s.prob}
                highlight={i === 0 ? "flagged" : undefined}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
