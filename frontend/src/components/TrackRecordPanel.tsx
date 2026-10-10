import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MarketReliability, TrackRecordByMarket, TrackRecordResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { MissesTable } from "./MissesTable";
import { GameweekResultsGrid } from "./GameweekResultsGrid";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
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

const MARKET_LABELS: Record<keyof TrackRecordByMarket, string> = {
  exact_score: "Exact score",
  match_result: "Match result",
  over_under_2_5: "Goals O/U 2.5",
  btts: "BTTS",
};

const NO_MARKET_DATA: MarketReliability = { pct_correct: null, n_resolved: 0 };
const DEFAULT_BY_MARKET: TrackRecordByMarket = {
  exact_score: NO_MARKET_DATA,
  match_result: NO_MARKET_DATA,
  over_under_2_5: NO_MARKET_DATA,
  btts: NO_MARKET_DATA,
};

export function TrackRecordPanel({ data }: { data: TrackRecordResponse }) {
  const { summary, biggest_upsets, gameweeks } = data;
  const [selected, setSelected] = useState<string | null>(null);
  // Defensive: an older public_snapshot.json built before this field
  // existed can still be live for a few minutes after a deploy (the
  // frontend ships instantly; the data it reads is regenerated on its own
  // schedule) -- confirmed live, this crashed the whole Data Hub page
  // rather than just missing a section.
  const byMarket = summary.by_market ?? DEFAULT_BY_MARKET;
  const nRebuilt = summary.n_rebuilt_fixtures ?? 0;
  // Optional for the same reason `by_market` is: a snapshot baked before the
  // field existed can still be live for a few minutes after a deploy, and the
  // frontend ships instantly while the data it reads is regenerated on its own
  // schedule. Absent means "this payload predates the field", not "zero picks".
//
// `summary.all_picks` is deliberately NOT read here any more. The backend RETAINS
// it, under its published name and still meaning every counted pick — which is
// now the same population as the headline — so the key does not disappear for
// any other consumer. Rendering it as a third card would be a duplicate of the
// headline under a longer label; the figure a reader actually needs beside the
// headline is `pre_kickoff`.
const preKickoff = summary.pre_kickoff ?? null;

  // Only a record with NOTHING at all falls back to the short message. A
  // record whose every pick was made after kickoff still has 50 graded picks
  // and a list worth showing, so it renders in full — the headline is the
  // record, and the pre-kickoff figure beside it is where the absence is read.
  if (summary.n_resolved_fixtures === 0 && nRebuilt === 0) {
    return (
      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Live track record
          <InfoTooltip text={GLOSSARY.trackRecordScore} align="left" />
        </h3>
        <p className="text-xs text-pl-text-faint">
          No scored predictions yet — this builds up automatically as fixtures are predicted and results come in.
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
          value={
            summary.pct_correct_overall === null
              ? "—"
              : `${pct(summary.pct_correct_overall)} (${Math.round(
                  summary.pct_correct_overall * summary.n_resolved_fixtures
                )}/${summary.n_resolved_fixtures})`
          }
          info={GLOSSARY.trackRecordScore}
        />
      </div>

      {/*
        B8's two figures, with their roles SWAPPED (spec
        2026-10-01-track-record-counts-every-pick). The headline above is every
        COUNTED pick, whenever it was made; the figure beside it is the subset
        made before kickoff, with its own n. That reversal is what stopped the
        headline emptying out on every model change — under the old rule a
        re-run on an already-played game stopped counting, and PL's shipped
        headline was `null` while 50 graded picks sat in the payload.

        Counting a late pick inflates the headline relative to a pure pre-game
        record, and that is the accepted cost of the decision. Publishing both
        figures is what makes it informed rather than hidden: the gap between
        them IS the size of the inflation, and the pre-kickoff figure is the
        honest read of live performance.
      */}
      {(preKickoff || nRebuilt > 0) && (
        <>
          {preKickoff && (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard
                label="Made before kickoff"
                value={
                  preKickoff.pct_correct_overall === null || preKickoff.n_resolved_fixtures === 0
                    ? "—"
                    : `${pct(preKickoff.pct_correct_overall)} (${Math.round(
                        preKickoff.pct_correct_overall * preKickoff.n_resolved_fixtures,
                      )}/${preKickoff.n_resolved_fixtures})`
                }
                info={GLOSSARY.trackRecordPreKickoff}
              />
            </div>
          )}
          <p className="text-xs text-pl-text-dim">
            {nRebuilt} pick{nRebuilt === 1 ? "" : "s"} made after kickoff{" "}
            {nRebuilt === 1 ? "is" : "are"} included in the record above.{" "}
            {preKickoff?.n_resolved_fixtures
              ? `The ${preKickoff.n_resolved_fixtures} made before kickoff are shown separately above.`
              : "None of them were made before kickoff, so the pre-kickoff figure is empty."}
          </p>
        </>
      )}

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Prediction reliability
          <InfoTooltip text={GLOSSARY.trackRecordByMarket} align="left" />
        </h3>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {(Object.keys(MARKET_LABELS) as (keyof typeof MARKET_LABELS)[]).map((market) => {
            const stat = byMarket[market];
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
              <span className="text-xs font-normal text-pl-text-dim">
                {group.pct_correct === null
                  ? "No pre-kickoff picks"
                  : `${pct(group.pct_correct)} correct (${Math.round(group.pct_correct * group.n_fixtures)}/${group.n_fixtures})`}
              </span>
            </h3>
            {group.pct_correct !== null && group.pct_correct_by_market && (
              <p className="mb-2 text-xs text-pl-text-faint">
                Exact score {pct(group.pct_correct_by_market.exact_score)} &middot; Goals O/U 2.5{" "}
                {pct(group.pct_correct_by_market.over_under_2_5)} &middot; BTTS {pct(group.pct_correct_by_market.btts)}
              </p>
            )}
            <GameweekResultsGrid results={group.fixtures} onSelect={setSelected} />
          </div>
        ))}
      </div>

      {selected && <FixtureModal eventId={selected} onClose={() => setSelected(null)} explain={api.explain} loadContext={api.loadContext} />}
    </div>
  );
}
