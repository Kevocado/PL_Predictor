import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RankingsResponse, TeamRanking } from "../types";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { teamColor } from "../lib/teamColors";
import { GLOSSARY } from "../lib/glossary";

function badgeFor(confidence: TeamRanking["confidence"], fitted: boolean): { label: string; tooltip: string } | null {
  if (confidence === "new") {
    return {
      label: "new",
      tooltip:
        "No matches played yet this season (or ever, within the model's loaded history) — this is a neutral, league-average placeholder rating, not a real fitted one. Updates to a real rating once the model is next retrained after they've played.",
    };
  }
  if (confidence === "limited") {
    return {
      label: "limited data",
      tooltip: "Only a few matches played so far — their attack/defence rating is based on limited evidence and can move a lot as more matches come in.",
    };
  }
  if (!fitted) {
    return {
      label: "placeholder rating",
      tooltip:
        "This team has a real match history, but not from a season the model has actually trained on (e.g. their only recent top-flight season is the one held out for validation) — so this is a neutral placeholder rather than a real fitted rating until the next retrain.",
    };
  }
  return null;
}

function RankingRow({ rank, team, attack, defence, maxMagnitude, confidence, fitted }: { rank: number; team: string; attack: number; defence: number; maxMagnitude: number; confidence: TeamRanking["confidence"]; fitted: boolean }) {
  const defensiveStrength = -defence; // more negative `defence` = stronger defensively
  const attackPct = (attack / maxMagnitude) * 100;
  const defencePct = (defensiveStrength / maxMagnitude) * 100;
  const badge = badgeFor(confidence, fitted);

  return (
    <div className="flex items-center gap-3 rounded-lg bg-pl-850/60 px-3 py-2">
      <span className="w-5 shrink-0 text-right text-xs font-semibold text-pl-text-faint">{rank}</span>
      <TeamBadge team={team} size="sm" />
      <span className="w-28 shrink-0 truncate text-sm text-pl-text">{team}</span>
      {badge && (
        <span className="flex shrink-0 items-center gap-1 rounded bg-pl-700/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-pl-text-dim">
          {badge.label}
          <InfoTooltip text={badge.tooltip} align="left" />
        </span>
      )}
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
            <RankingRow key={r.team} rank={i + 1} team={r.team} attack={r.attack} defence={r.defence} maxMagnitude={maxMagnitude} confidence={r.confidence} fitted={r.fitted} />
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
