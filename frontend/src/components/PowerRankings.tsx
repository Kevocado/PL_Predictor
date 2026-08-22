import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RankingsResponse } from "../types";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { teamColor } from "../lib/teamColors";
import { GLOSSARY } from "../lib/glossary";

function RankingRow({ rank, team, attack, defence, maxMagnitude }: { rank: number; team: string; attack: number; defence: number; maxMagnitude: number }) {
  const defensiveStrength = -defence; // more negative `defence` = stronger defensively
  const attackPct = (attack / maxMagnitude) * 100;
  const defencePct = (defensiveStrength / maxMagnitude) * 100;

  return (
    <div className="flex items-center gap-3 rounded-lg bg-pl-850/60 px-3 py-2">
      <span className="w-5 shrink-0 text-right text-xs font-semibold text-pl-text-faint">{rank}</span>
      <TeamBadge team={team} size="sm" />
      <span className="w-28 shrink-0 truncate text-sm text-pl-text">{team}</span>
      <div className="flex flex-1 items-center gap-2">
        <div className="flex h-2 flex-1 justify-end overflow-hidden rounded-l-full bg-pl-850">
          <div className="h-full rounded-l-full bg-win" style={{ width: `${Math.max(attackPct, 2)}%` }} />
        </div>
        <div className="flex h-2 flex-1 overflow-hidden rounded-r-full bg-pl-850">
          <div className="h-full rounded-r-full bg-pl-pink" style={{ width: `${Math.max(defencePct, 2)}%` }} />
        </div>
      </div>
    </div>
  );
}

export function PowerRankings({ data }: { data: RankingsResponse }) {
  const maxMagnitude = Math.max(...data.rankings.flatMap((r) => [r.attack, -r.defence]), 0.1);
  const teams = Object.keys(data.ratings_history);

  const chartData: Record<string, number | string>[] = [];
  const dateSet = new Set<string>();
  for (const points of Object.values(data.ratings_history)) {
    for (const p of points) dateSet.add(p.date);
  }
  const dates = [...dateSet].sort();
  for (const date of dates) {
    const row: Record<string, number | string> = { date };
    for (const [team, points] of Object.entries(data.ratings_history)) {
      const match = points.find((p) => p.date === date);
      if (match) row[team] = Math.round(match.elo);
    }
    chartData.push(row);
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h3 className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Power rankings
          <InfoTooltip text={GLOSSARY.powerRankings} align="left" />
        </h3>
        <div className="mb-2 flex items-center gap-4 text-[11px] text-pl-text-faint">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-win" /> Attack
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-pl-pink" /> Defence
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {data.rankings.map((r, i) => (
            <RankingRow key={r.team} rank={i + 1} team={r.team} attack={r.attack} defence={r.defence} maxMagnitude={maxMagnitude} />
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Elo rating this season
          <InfoTooltip text={GLOSSARY.ratingsTrend} align="left" />
        </h3>
        {teams.length === 0 ? (
          <p className="text-xs text-pl-text-faint">
            No matches played yet this season — the trend will fill in once the season kicks off.
          </p>
        ) : (
          <div className="clip-corner-lg h-96 rounded-xl border border-pl-border bg-pl-850/70 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 10 }} />
                <YAxis domain={["auto", "auto"]} tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                  labelStyle={{ color: "var(--color-pl-text-faint)" }}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {teams.map((team) => (
                  <Line key={team} type="monotone" dataKey={team} stroke={teamColor(team)} strokeWidth={1.5} dot={false} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
