import { useState } from "react";
import { api } from "../api/client";
import type { BettingValidationBreakdown, WalkForwardBettingResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";

function StatCard({ label, value, positive, info }: { label: string; value: string; positive?: boolean; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className={`mt-1 text-2xl font-bold ${positive === undefined ? "text-pl-text" : positive ? "text-win" : "text-loss"}`}>{value}</div>
    </div>
  );
}

function BreakdownTable({ title, rows }: { title: string; rows: BettingValidationBreakdown[] }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <h3 className="text-sm font-semibold text-pl-text">{title}</h3>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[340px] text-left text-xs">
          <thead className="border-b border-pl-border text-[10px] uppercase tracking-wide text-pl-text-faint">
            <tr><th className="pb-2 font-semibold">Group</th><th className="pb-2 font-semibold">Bets</th><th className="pb-2 font-semibold">Win rate</th><th className="pb-2 font-semibold">Yield</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-b border-pl-border/60 last:border-0">
                <td className="py-2 text-pl-text">{row.label}</td>
                <td className="py-2 text-pl-text-dim">{row.bets}</td>
                <td className="py-2 text-pl-text-dim">{row.win_rate === null ? "—" : `${row.win_rate.toFixed(1)}%`}</td>
                <td className={`py-2 font-semibold ${(row.yield ?? 0) >= 0 ? "text-win" : "text-loss"}`}>{row.yield === null ? "—" : `${row.yield >= 0 ? "+" : ""}${row.yield.toFixed(1)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function WalkForwardBettingPanel() {
  const [data, setData] = useState<WalkForwardBettingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setLoading(true);
    setError(null);
    api.walkForwardValueBetValidation().then(setData).catch((error) => setError(error.message)).finally(() => setLoading(false));
  };

  const summary = data?.summary;
  const interval = summary?.yield_ci_95;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="max-w-2xl text-xs text-pl-text-faint">Each season is tested only after training on earlier seasons. Yield uses one unit per pick; the 95% interval shows how uncertain the combined historical yield remains. This is a robustness check, not a forecast.</p>
        <button onClick={run} disabled={loading} className="rounded-lg bg-pl-pink px-4 py-2 text-sm font-semibold text-white transition hover:bg-pl-pink-soft disabled:opacity-50">{loading ? "Testing…" : "Run walk-forward test"}</button>
      </div>
      {error && <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>}
      {data && summary && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Bets" value={String(summary.bets)} />
            <StatCard label="Win rate" value={summary.win_rate === null ? "—" : `${summary.win_rate.toFixed(1)}%`} />
            <StatCard label="Yield" value={summary.yield === null ? "—" : `${summary.yield >= 0 ? "+" : ""}${summary.yield.toFixed(1)}%`} positive={(summary.yield ?? 0) >= 0} info="Net return per one unit staked, before any real-world price movement or limits." />
            <StatCard label="95% yield range" value={interval ? `${interval[0].toFixed(1)}% to ${interval[1].toFixed(1)}%` : "—"} positive={interval ? interval[0] > 0 : undefined} info="Bootstrap interval from the historical picks. If it crosses zero, the data does not yet establish a positive edge." />
          </div>
          <BreakdownTable title="Season-by-season folds" rows={data.folds.map(({ season, train_matches, ...row }) => ({ ...row, label: `${season} (${train_matches} train)` }))} />
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2"><BreakdownTable title="By market" rows={data.by_market} /><BreakdownTable title="By odds band" rows={data.by_odds_band} /></div>
        </>
      )}
    </div>
  );
}
