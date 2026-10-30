import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { BacktestResponse } from "../types";
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
  over_2_5: "Over 2.5",
  under_2_5: "Under 2.5",
};

export function BacktestPanel() {
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [staking, setStaking] = useState<"kelly" | "flat">("kelly");

  const run = (mode: "kelly" | "flat" = staking) => {
    setLoading(true);
    setError(null);
    api
      .backtest(mode)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const selectStaking = (mode: "kelly" | "flat") => {
    setStaking(mode);
    if (result) run(mode); // already ran once — keep the comparison live as the toggle changes
  };

  const chartData = result?.bankroll_curve.map((v, i) => ({ bet: i + 1, bankroll: v })) ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <p className="max-w-xl text-xs text-pr-text-faint">
          Replays the held-out season using archived Bet365 closing prices. It applies the same 5-point de-vigged
          edge, single-best-selection, and odds-cap rules as the live recommendation card across match result and
          O/U 2.5 goals. This is historical validation, not the live ledger.
        </p>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-pr-panel/70 p-1 text-xs">
            <button
              onClick={() => selectStaking("kelly")}
              className={`rounded-md px-2.5 py-1 font-semibold transition ${
                staking === "kelly" ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-faint hover:text-pr-text"
              }`}
            >
              Kelly stakes
            </button>
            <button
              onClick={() => selectStaking("flat")}
              className={`rounded-md px-2.5 py-1 font-semibold transition ${
                staking === "flat" ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-faint hover:text-pr-text"
              }`}
            >
              Flat stakes
            </button>
          </div>
          <button
            onClick={() => run()}
            disabled={loading}
            className="rounded-lg bg-pr-accent px-4 py-2 text-sm font-semibold text-pr-accent-ink transition hover:bg-pr-accent disabled:opacity-50"
          >
            {loading ? "Running…" : "Run historical replay"}
          </button>
        </div>
      </div>

      <p className="text-xs text-pr-text-faint">
        {staking === "kelly"
          ? "Kelly stakes size each bet by how confident the model is (a tenth-Kelly fraction — see tooltip), instead of betting the same amount every time. This is the more honest simulation: it reveals how much a real bettor's bankroll would swing if they actually trusted the model's stated probabilities enough to size up on them."
          : "Flat stakes bet the same fixed amount on every qualifying edge, regardless of how confident the model is. Simpler, but it hides how risky following the model's stated confidence would actually be."}
      </p>

      {error && <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>}

      {result && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Total bets" value={String(result.results["Total Bets"])} />
            <StatCard
              label="Win rate"
              value={`${result.results["Successful Bet %"].toFixed(1)}%`}
            />
            <StatCard
              label="Profit"
              value={`${result.results.Profit >= 0 ? "+" : ""}${result.results.Profit.toFixed(2)}`}
              positive={result.results.Profit >= 0}
            />
            <StatCard
              label="ROI"
              value={`${result.results.ROI >= 0 ? "+" : ""}${result.results.ROI.toFixed(1)}%`}
              positive={result.results.ROI >= 0}
              info={GLOSSARY.roi}
            />
          </div>

          <div className="clip-corner rounded-pr border border-pr-rule bg-pr-panel/70 p-4">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-sm font-semibold text-pr-text">Historical picks</h3>
              <span className="text-xs text-pr-text-faint">{result.season ?? "Held-out season"} · archived Bet365 closing prices</span>
            </div>
            {result.selections.length === 0 ? (
              <p className="mt-3 text-xs text-pr-text-faint">No historical fixture cleared the live value-bet rules.</p>
            ) : (
              <div className="mt-3 max-h-80 overflow-auto">
                <table className="w-full min-w-[680px] text-left text-xs">
                  <thead className="sticky top-0 border-b border-pr-rule bg-pr-panel/95 text-xs uppercase tracking-wide text-pr-text-faint">
                    <tr>
                      <th className="pb-2 font-semibold">Date / final score</th>
                      <th className="pb-2 font-semibold">Pick</th>
                      <th className="pb-2 font-semibold">Price</th>
                      <th className="pb-2 font-semibold">Model</th>
                      <th className="pb-2 font-semibold">Edge</th>
                      <th className="pb-2 font-semibold">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.selections.map((bet) => (
                      <tr key={`${bet.date}-${bet.fixture}-${bet.selection}`} className="border-b border-pr-rule/60 last:border-0">
                        <td className="py-2 text-pr-text"><span className="mr-2 text-pr-text-faint">{bet.date}</span>{bet.fixture}</td>
                        <td className="py-2 text-pr-text-dim">{SELECTION_LABELS[bet.selection]}</td>
                        <td className="py-2 text-pr-text-dim">{americanOdds(bet.price)}</td>
                        <td className="py-2 text-pr-text-dim">{(bet.model_probability * 100).toFixed(1)}%</td>
                        <td className="py-2 font-semibold text-pr-lean">+{(bet.edge * 100).toFixed(1)}%</td>
                        <td className={`py-2 font-semibold ${bet.won ? "text-win" : "text-loss"}`}>{bet.won ? "Won" : "Lost"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

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
        </>
      )}
    </div>
  );
}
