import type { GameweekResult } from "../types";

export function RecentResultsTable({ results }: { results: GameweekResult[] }) {
  if (results.length === 0) {
    return (
      <p className="text-xs text-pl-text-faint">
        No resolved fixtures yet — this fills in automatically as matches kick off and finish.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-pl-border">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-pl-border bg-pl-850/70 text-left text-[11px] uppercase tracking-wide text-pl-text-faint">
            <th className="px-3 py-2">Fixture</th>
            <th className="px-3 py-2">Predicted vs. actual</th>
            <th className="px-3 py-2 text-right">H / D / A</th>
            <th className="px-3 py-2 text-right">Called it?</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={`${r.team_home}-${r.team_away}-${r.commence_time}`} className="border-b border-pl-border/60 bg-pl-850/40">
              <td className="px-3 py-2 text-pl-text-faint">
                <div className="flex items-center gap-1.5 text-pl-text">
                  {r.team_home} v {r.team_away}
                  {r.backfilled && (
                    <span
                      title="This match had already finished before the app started tracking it live — the prediction shown is what the model would have said, computed the same way as any live prediction, just not literally captured before kickoff."
                      className="rounded border border-pl-border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-pl-text-faint"
                    >
                      Backfilled
                    </span>
                  )}
                </div>
                <div className="text-[10px]">{new Date(r.commence_time).toLocaleDateString(undefined, { day: "numeric", month: "short" })}</div>
              </td>
              <td className="px-3 py-2 font-mono font-semibold text-pl-text">
                {r.predicted_scoreline ?? "?"} <span className="text-pl-text-faint">→</span>{" "}
                {r.actual_goals_home ?? "?"}-{r.actual_goals_away ?? "?"}
              </td>
              <td className="px-3 py-2 text-right text-[11px] text-pl-text-faint">
                {(r.predicted_home_win * 100).toFixed(0)}% / {(r.predicted_draw * 100).toFixed(0)}% /{" "}
                {(r.predicted_away_win * 100).toFixed(0)}%
              </td>
              <td className={`px-3 py-2 text-right font-semibold ${r.hit ? "text-win" : "text-loss"}`}>{r.hit ? "✓" : "✗"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
