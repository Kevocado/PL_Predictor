import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { ValueBetTrackRecordResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, positive, info }: { label: string; value: string; positive?: boolean; info?: string }) {
  return (
    <div className="clip-corner rounded-pr border border-pr-rule bg-pr-panel/70 p-4">
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pr-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className={`mt-1 font-display text-3xl font-semibold tracking-wide ${positive === undefined ? "text-pr-text" : positive ? "text-win" : "text-loss"}`}>
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
        <p className="max-w-xl text-xs text-pr-text-faint">
          Every fixture actually flagged "Value bet" in the app (see the Fixtures page) gets snapshotted the moment
          it's first flagged — including the live bookmaker price at that instant — and reconciled once the match
          finishes. This is the real live record, not a simulation: does following the app's own recommendations
          over time actually make money?
        </p>
        <div className="flex shrink-0 items-center gap-1 rounded-lg bg-pr-panel/70 p-1 text-xs">
          <button
            onClick={() => setStaking("kelly")}
            className={`rounded-md px-2.5 py-1 font-semibold transition ${
              staking === "kelly" ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-faint hover:text-pr-text"
            }`}
          >
            Kelly stakes
          </button>
          <button
            onClick={() => setStaking("flat")}
            className={`rounded-md px-2.5 py-1 font-semibold transition ${
              staking === "flat" ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-faint hover:text-pr-text"
            }`}
          >
            Flat stakes
          </button>
        </div>
      </div>

      {error && <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>}

      {loading && !data && <div className="py-8 text-center text-xs text-pr-text-faint">Loading live value-bet record…</div>}

      {data && data.n_flagged === 0 && (
        <p className="text-xs text-pr-text-faint">
          No fixtures have been flagged as a value bet yet — this builds up automatically as live odds come in and
          the model finds edges above its confidence-adjusted threshold.
        </p>
      )}

      {data && data.n_flagged > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Flagged" value={String(data.n_flagged)} info="Every fixture/market ever flagged as a value bet, resolved or not." />
            <StatCard label="Confirmed W-L" value={`${data.confirmed_wins}-${data.confirmed_losses}`} info="Only final scores confirmed by the result feed count here." />
            <StatCard label="Pending" value={String(data.n_pending)} info="Flagged fixtures that haven't kicked off / finished yet — still open, so they are not yet settled in the results below." />
            <StatCard label="Win rate" value={data.confirmed_win_rate === null ? "—" : `${data.confirmed_win_rate.toFixed(1)}%`} info="Actual wins divided by every confirmed value-bet result, independent of staking or odds caps." />
            <StatCard
              label="ROI"
              value={results ? `${results.ROI >= 0 ? "+" : ""}${results.ROI.toFixed(1)}%` : "—"}
              positive={results ? results.ROI >= 0 : undefined}
              info={`${GLOSSARY.roi} Uses the selected staking mode and excludes prices beyond the app's risk cap.`}
            />
            <StatCard
              label="Avg CLV"
              value={data.average_clv_pct === null ? "—" : `${data.average_clv_pct >= 0 ? "+" : ""}${data.average_clv_pct.toFixed(1)}%`}
              positive={data.average_clv_pct === null ? undefined : data.average_clv_pct >= 0}
              info={`${GLOSSARY.clv} Averaged over ${data.n_with_closing_line} bet${data.n_with_closing_line === 1 ? "" : "s"} where a closing price was actually captured before kickoff.`}
            />
          </div>

          <div className="clip-corner rounded-pr border border-pr-rule bg-pr-panel/70 p-4">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-sm font-semibold text-pr-text">Confirmed results</h3>
              <span className="text-xs text-pr-text-faint">Final scores from the named result source</span>
            </div>
            {data.confirmed_bets.length === 0 ? (
              <p className="mt-3 text-xs text-pr-text-faint">No flagged fixture has a confirmed final result yet.</p>
            ) : (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[620px] text-left text-xs">
                  <thead className="border-b border-pr-rule text-xs uppercase tracking-wide text-pr-text-faint">
                    <tr>
                      <th className="pb-2 font-semibold">Fixture / final score</th>
                      <th className="pb-2 font-semibold">Pick</th>
                      <th className="pb-2 font-semibold">Price</th>
                      <th className="pb-2 font-semibold">Edge</th>
                      <th className="pb-2 font-semibold">CLV</th>
                      <th className="pb-2 font-semibold">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.confirmed_bets.slice(0, 12).map((bet, index) => (
                      <tr key={`${bet.fixture}-${bet.selection}-${index}`} className="border-b border-pr-rule/60 last:border-0">
                        <td className="py-2 text-pr-text">
                          {bet.fixture}
                          <span className="ml-2 text-xs text-pr-text-faint">{bet.result_source}</span>
                        </td>
                        <td className="py-2 text-pr-text-dim">{SELECTION_LABELS[bet.selection] ?? bet.selection}</td>
                        <td className="py-2 text-pr-text-dim">{americanOdds(bet.price)}</td>
                        <td className="py-2 font-semibold text-pr-lean">+{(bet.edge * 100).toFixed(1)}%</td>
                        <td className={`py-2 font-semibold ${bet.clv_pct === null ? "text-pr-text-faint" : bet.clv_pct >= 0 ? "text-win" : "text-loss"}`}>
                          {bet.clv_pct === null ? "—" : `${bet.clv_pct >= 0 ? "+" : ""}${bet.clv_pct.toFixed(1)}%`}
                        </td>
                        <td className={`py-2 font-semibold ${bet.won ? "text-win" : "text-loss"}`}>{bet.won ? "Won" : "Lost"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {results && results["Total Bets"] > 0 ? (
            <div className="clip-corner-lg h-72 rounded-pr border border-pr-rule bg-pr-panel/70 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--color-pr-rule)" strokeDasharray="3 3" />
                  <XAxis dataKey="bet" tick={{ fill: "var(--color-pr-text-faint)", fontSize: 11 }} />
                  <YAxis tick={{ fill: "var(--color-pr-text-faint)", fontSize: 11 }} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{ background: "var(--color-pr-stage)", border: "1px solid var(--color-pr-rule)", borderRadius: 8 }}
                    labelStyle={{ color: "var(--color-pr-text-faint)" }}
                  />
                  <ReferenceLine y={100} stroke="var(--color-pr-text-faint)" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="bankroll" stroke="var(--color-pr-accent)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-xs text-pr-text-faint">
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
