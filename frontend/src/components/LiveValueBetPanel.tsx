import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { ValueBetTrackRecordResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, positive, info }: { label: string; value: string; positive?: boolean; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className={`mt-1 text-2xl font-bold ${positive === undefined ? "text-pl-text" : positive ? "text-win" : "text-loss"}`}>
        {value}
      </div>
    </div>
  );
}

function americanOdds(decimalOdds: number) {
  return decimalOdds >= 2 ? `+${Math.round((decimalOdds - 1) * 100)}` : `${Math.round(-100 / (decimalOdds - 1))}`;
}

const SELECTION_LABELS: Record<string, string> = {
  home_win: "Home win",
  draw: "Draw",
  away_win: "Away win",
  over: "Over 2.5",
  under: "Under 2.5",
};

export function LiveValueBetPanel() {
  const [data, setData] = useState<ValueBetTrackRecordResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [staking, setStaking] = useState<"kelly" | "flat">("kelly");

  const load = (mode: "kelly" | "flat" = staking) => {
    setLoading(true);
    setError(null);
    api
      .valueBetTrackRecord(mode)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => load(staking), [staking]);

  const chartData = data?.bankroll_curve.map((v, i) => ({ bet: i + 1, bankroll: v })) ?? [];
  const results = data?.results;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <p className="max-w-xl text-xs text-pl-text-faint">
          Every fixture actually flagged "Value bet" in the app (see the Fixtures page) gets snapshotted the moment
          it's first flagged — including the live bookmaker price at that instant — and reconciled once the match
          finishes. This is the real live record, not a simulation: does following the app's own recommendations
          over time actually make money?
        </p>
        <div className="flex shrink-0 items-center gap-1 rounded-lg bg-pl-850/70 p-1 text-xs">
          <button
            onClick={() => setStaking("kelly")}
            className={`rounded-md px-2.5 py-1 font-semibold transition ${
              staking === "kelly" ? "bg-pl-pink text-white" : "text-pl-text-faint hover:text-pl-text"
            }`}
          >
            Kelly stakes
          </button>
          <button
            onClick={() => setStaking("flat")}
            className={`rounded-md px-2.5 py-1 font-semibold transition ${
              staking === "flat" ? "bg-pl-pink text-white" : "text-pl-text-faint hover:text-pl-text"
            }`}
          >
            Flat stakes
          </button>
        </div>
      </div>

      {error && <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>}

      {loading && !data && <div className="py-8 text-center text-xs text-pl-text-faint">Loading live value-bet record…</div>}

      {data && data.n_flagged === 0 && (
        <p className="text-xs text-pl-text-faint">
          No fixtures have been flagged as a value bet yet — this builds up automatically as live odds come in and
          the model finds edges above its confidence-adjusted threshold.
        </p>
      )}

      {data && data.n_flagged > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">
            <StatCard label="Flagged" value={String(data.n_flagged)} info="Every fixture/market ever flagged as a value bet, resolved or not." />
            <StatCard label="Confirmed W-L" value={`${data.confirmed_wins}-${data.confirmed_losses}`} info="Only final scores confirmed by the result feed count here." />
            <StatCard label="Pending" value={String(data.n_pending)} info="Flagged fixtures that haven't kicked off / finished yet — open positions, not counted in the results below." />
            <StatCard label="Win rate" value={data.confirmed_win_rate === null ? "—" : `${data.confirmed_win_rate.toFixed(1)}%`} info="Actual wins divided by every confirmed value-bet result, independent of staking or odds caps." />
            <StatCard
              label="ROI"
              value={results ? `${results.ROI >= 0 ? "+" : ""}${results.ROI.toFixed(1)}%` : "—"}
              positive={results ? results.ROI >= 0 : undefined}
              info={`${GLOSSARY.roi} Uses the selected staking mode and excludes prices beyond the app's risk cap.`}
            />
          </div>

          <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-sm font-semibold text-pl-text">Confirmed results</h3>
              <span className="text-[11px] text-pl-text-faint">Final scores from the named result source</span>
            </div>
            {data.confirmed_bets.length === 0 ? (
              <p className="mt-3 text-xs text-pl-text-faint">No flagged fixture has a confirmed final result yet.</p>
            ) : (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[620px] text-left text-xs">
                  <thead className="border-b border-pl-border text-[10px] uppercase tracking-wide text-pl-text-faint">
                    <tr>
                      <th className="pb-2 font-semibold">Fixture / final score</th>
                      <th className="pb-2 font-semibold">Pick</th>
                      <th className="pb-2 font-semibold">Price</th>
                      <th className="pb-2 font-semibold">Edge</th>
                      <th className="pb-2 font-semibold">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.confirmed_bets.slice(0, 12).map((bet, index) => (
                      <tr key={`${bet.fixture}-${bet.selection}-${index}`} className="border-b border-pl-border/60 last:border-0">
                        <td className="py-2 text-pl-text">
                          {bet.fixture}
                          <span className="ml-2 text-[10px] text-pl-text-faint">{bet.result_source}</span>
                        </td>
                        <td className="py-2 text-pl-text-dim">{SELECTION_LABELS[bet.selection] ?? bet.selection}</td>
                        <td className="py-2 text-pl-text-dim">{americanOdds(bet.price)}</td>
                        <td className="py-2 font-semibold text-pl-cyan">+{(bet.edge * 100).toFixed(1)}%</td>
                        <td className={`py-2 font-semibold ${bet.won ? "text-win" : "text-loss"}`}>{bet.won ? "Won" : "Lost"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {results && results["Total Bets"] > 0 ? (
            <div className="clip-corner-lg h-72 rounded-xl border border-pl-border bg-pl-850/70 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
                  <XAxis dataKey="bet" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} />
                  <YAxis tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                    labelStyle={{ color: "var(--color-pl-text-faint)" }}
                  />
                  <ReferenceLine y={100} stroke="var(--color-pl-text-faint)" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="bankroll" stroke="var(--color-pl-pink)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-xs text-pl-text-faint">
              {data.n_resolved === 0
                ? "No flagged fixtures have finished yet — check back once the first ones kick off and resolve."
                : "No qualifying bets under this staking mode yet (e.g. every resolved flag was priced beyond the 6/1 cutoff)."}
            </p>
          )}
        </>
      )}
    </div>
  );
}
