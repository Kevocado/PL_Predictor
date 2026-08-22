import type { MissedPrediction } from "../types";

const MARKET_LABELS: Record<string, string> = {
  "1x2": "Result",
  totals_2_5: "Goals O/U 2.5",
  btts: "BTTS",
};

const OUTCOME_LABELS: Record<string, string> = {
  home_win: "Home",
  draw: "Draw",
  away_win: "Away",
  over: "Over",
  under: "Under",
  yes: "Yes",
};

export function MissesTable({ misses }: { misses: MissedPrediction[] }) {
  if (misses.length === 0) {
    return <p className="text-xs text-pl-text-faint">No resolved predictions to review yet.</p>;
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-pl-border">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-pl-border bg-pl-850/70 text-left text-[11px] uppercase tracking-wide text-pl-text-faint">
            <th className="px-3 py-2">Fixture</th>
            <th className="px-3 py-2">Market</th>
            <th className="px-3 py-2 text-right">Predicted</th>
            <th className="px-3 py-2 text-right">Actual</th>
          </tr>
        </thead>
        <tbody>
          {misses.map((m, i) => (
            <tr key={i} className="border-b border-pl-border/60 bg-pl-850/40">
              <td className="px-3 py-2 text-pl-text">
                {m.team_home} v {m.team_away}
              </td>
              <td className="px-3 py-2 text-pl-text-dim">
                {MARKET_LABELS[m.market] ?? m.market}: {OUTCOME_LABELS[m.outcome_name] ?? m.outcome_name}
              </td>
              <td className="px-3 py-2 text-right font-semibold text-pl-text">{(m.predicted_prob * 100).toFixed(0)}%</td>
              <td className={`px-3 py-2 text-right font-semibold ${m.actual_outcome ? "text-win" : "text-loss"}`}>
                {m.actual_outcome ? "Happened" : "Didn't happen"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
