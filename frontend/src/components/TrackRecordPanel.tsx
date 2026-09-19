import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TrackRecordResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { MissesTable } from "./MissesTable";
import { GameweekResultsGrid } from "./GameweekResultsGrid";
import { FixtureModal } from "./FixtureModal";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className="mt-1 font-display text-3xl font-semibold tracking-wide text-pl-text">{value}</div>
    </div>
  );
}

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(0)}%`;
}

const MARKET_LABELS: Record<keyof TrackRecordResponse["summary"]["by_market"], string> = {
  exact_score: "Exact score",
  match_result: "Match result",
  over_under_2_5: "Goals O/U 2.5",
  btts: "BTTS",
};

export function TrackRecordPanel({ data }: { data: TrackRecordResponse }) {
  const { summary, biggest_upsets, gameweeks } = data;
  const [selected, setSelected] = useState<string | null>(null);

  if (summary.n_resolved_fixtures === 0) {
    return (
      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Live track record
          <InfoTooltip text={GLOSSARY.trackRecordScore} align="left" />
        </h3>
        <p className="text-xs text-pl-text-faint">
          No resolved predictions yet — this builds up automatically as fixtures kick off and results come in.
          Predictions are snapshotted the moment a fixture first appears in the app, before it's played.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          label={summary.current_gameweek ? `Gameweek ${summary.current_gameweek}` : "This gameweek"}
          value={
            summary.pct_correct_current_gameweek === null
              ? "—"
              : `${pct(summary.pct_correct_current_gameweek)} (${Math.round(
                  summary.pct_correct_current_gameweek * summary.n_fixtures_current_gameweek
                )}/${summary.n_fixtures_current_gameweek})`
          }
          info={GLOSSARY.trackRecordScore}
        />
        <StatCard
          label="Correct overall"
          value={`${pct(summary.pct_correct_overall)} (${Math.round(
            (summary.pct_correct_overall ?? 0) * summary.n_resolved_fixtures
          )}/${summary.n_resolved_fixtures})`}
          info={GLOSSARY.trackRecordScore}
        />
      </div>

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Prediction reliability
          <InfoTooltip text={GLOSSARY.trackRecordByMarket} align="left" />
        </h3>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {(Object.keys(MARKET_LABELS) as (keyof typeof MARKET_LABELS)[]).map((market) => {
            const stat = summary.by_market[market];
            return (
              <StatCard
                key={market}
                label={MARKET_LABELS[market]}
                value={stat.n_resolved === 0 ? "—" : `${pct(stat.pct_correct)} (${Math.round((stat.pct_correct ?? 0) * stat.n_resolved)}/${stat.n_resolved})`}
              />
            );
          })}
        </div>
      </div>

      {summary.gameweek_trend.length > 1 && (
        <div className="clip-corner-lg h-64 rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={summary.gameweek_trend} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
              <XAxis
                dataKey="gameweek"
                tick={{ fill: "var(--color-pl-text-faint)", fontSize: 10 }}
                tickFormatter={(gw) => `GW${gw}`}
              />
              <YAxis
                tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                domain={[0, 1]}
              />
              <Tooltip
                contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                labelStyle={{ color: "var(--color-pl-text-faint)" }}
                labelFormatter={(gw) => `Gameweek ${gw}`}
                formatter={(v) => [`${(Number(v) * 100).toFixed(0)}%`, "Correct"]}
              />
              <Bar dataKey="pct_correct" fill="var(--color-pl-pink)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Biggest upset
          <InfoTooltip text={GLOSSARY.biggestMisses} align="left" />
        </h3>
        <MissesTable misses={biggest_upsets.slice(0, 3)} />
      </div>

      <div className="flex flex-col gap-5">
        {gameweeks.map((group) => (
          <div key={group.gameweek ?? "none"}>
            <h3 className="mb-1 flex items-center gap-2 text-sm font-semibold text-pl-text-dim">
              {group.gameweek ? `Gameweek ${group.gameweek}` : "No gameweek data"}
              <span className="text-xs font-normal text-pl-text-faint">
                {pct(group.pct_correct)} correct ({Math.round(group.pct_correct * group.n_fixtures)}/{group.n_fixtures})
              </span>
            </h3>
            <p className="mb-2 text-[11px] text-pl-text-faint">
              Exact score {pct(group.pct_correct_by_market.exact_score)} &middot; Goals O/U 2.5{" "}
              {pct(group.pct_correct_by_market.over_under_2_5)} &middot; BTTS {pct(group.pct_correct_by_market.btts)}
            </p>
            <GameweekResultsGrid results={group.fixtures} onSelect={setSelected} />
          </div>
        ))}
      </div>

      {selected && <FixtureModal eventId={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
