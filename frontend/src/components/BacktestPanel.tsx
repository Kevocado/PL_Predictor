import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { BacktestResponse } from "../types";
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
        <p className="max-w-xl text-xs text-pl-text-faint">
          Simulates betting the held-out season whenever the model's probability beats the raw bookmaker-implied
          probability by 5%+ — one bet per fixture (whichever side has the biggest edge), skipping odds beyond 6/1
          where the model's calibration is least trustworthy. A strongly positive ROI here is a red flag for
          overfitting, not a target — expect roughly break-even to slightly negative.
        </p>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-pl-850/70 p-1 text-xs">
            <button
              onClick={() => selectStaking("kelly")}
              className={`rounded-md px-2.5 py-1 font-semibold transition ${
                staking === "kelly" ? "bg-pl-pink text-white" : "text-pl-text-faint hover:text-pl-text"
              }`}
            >
              Kelly stakes
            </button>
            <button
              onClick={() => selectStaking("flat")}
              className={`rounded-md px-2.5 py-1 font-semibold transition ${
                staking === "flat" ? "bg-pl-pink text-white" : "text-pl-text-faint hover:text-pl-text"
              }`}
            >
              Flat stakes
            </button>
          </div>
          <button
            onClick={() => run()}
            disabled={loading}
            className="rounded-lg bg-pl-pink px-4 py-2 text-sm font-semibold text-white transition hover:bg-pl-pink-soft disabled:opacity-50"
          >
            {loading ? "Running…" : "Run backtest"}
          </button>
        </div>
      </div>

      <p className="text-[11px] text-pl-text-faint">
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
        </>
      )}
    </div>
  );
}
