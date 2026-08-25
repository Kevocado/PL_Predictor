import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RankingsResponse, TeamRanking } from "../types";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { teamColor } from "../lib/teamColors";
import { GLOSSARY } from "../lib/glossary";

function badgeFor(confidence: TeamRanking["confidence"], fitted: boolean): { label: string; tooltip: string } | null {
  if (confidence === "preseason") {
    return {
      label: "preseason prior",
      tooltip:
        "This ranking is deliberately anchored to historical team strength before current-season results are allowed to influence it. It avoids treating one opening-week result as decisive.",
    };
  }
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
      tooltip:
        "Only a few current-season matches have been played. The live ranking blends their online form with the pre-season prior, so it can move as more results arrive.",
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

function MetricBar({ score, color, direction }: { score: number; color: string; direction: "left" | "right" }) {
  const position = direction === "left" ? "right-0" : "left-0";

  return (
    <div className="relative flex h-8 items-center justify-center overflow-hidden bg-pl-800">
      <div className={`absolute inset-y-0 ${position} ${color}`} style={{ width: `${score}%` }} />
      <span className="relative z-10 text-xs font-semibold tabular-nums text-white drop-shadow-sm">{Math.round(score)}</span>
    </div>
  );
}

function RankingRow({ rank, team, attackScore, defenceScore, confidence, fitted }: { rank: number; team: string; attackScore: number; defenceScore: number; confidence: TeamRanking["confidence"]; fitted: boolean }) {
  const badge = badgeFor(confidence, fitted);

  return (
    <div className="grid grid-cols-[1.25rem_1.5rem_minmax(6rem,1fr)_auto_minmax(5rem,1fr)_1px_minmax(5rem,1fr)] items-center gap-2 rounded-lg bg-pl-850/60 px-3 py-2">
      <span className="text-right text-xs font-semibold text-pl-text-faint">{rank}</span>
      <TeamBadge team={team} size="sm" />
      <span className="truncate text-sm text-pl-text">{team}</span>
      {badge && (
        <span className="flex items-center gap-1 rounded bg-pl-700/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-pl-text-dim">
          {badge.label}
          <InfoTooltip text={badge.tooltip} align="left" />
        </span>
      )}
      <MetricBar score={attackScore} color="bg-win" direction="left" />
      <span className="h-8 bg-pl-text-faint/60" aria-hidden="true" />
      <MetricBar score={defenceScore} color="bg-pl-pink" direction="right" />
    </div>
  );
}

export function PowerRankings({ data }: { data: RankingsResponse }) {
  const clampScore = (value: number) => Math.min(100, Math.max(0, value));
  const attackScore = (value: number) => clampScore(value * 50);
  const defenceScore = (value: number) => clampScore(-value * 62.5);
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
        <div className="mb-2 grid grid-cols-[1.25rem_1.5rem_minmax(6rem,1fr)_auto_minmax(5rem,1fr)_1px_minmax(5rem,1fr)] gap-2 px-3 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
          <span className="col-start-5 flex items-center justify-center gap-1 text-center">
            Attack
            <InfoTooltip text="Fixed 0–100 conversion of the model's attacking-strength value. 50 is typical league-level attack; higher means stronger." />
          </span>
          <span className="col-start-7 flex items-center justify-center gap-1 text-center">
            Defence
            <InfoTooltip text="Fixed 0–100 conversion of the model's defensive-strength value. 50 is typical league-level defence; higher means stronger." />
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {data.rankings.map((r, i) => (
            <RankingRow
              key={r.team}
              rank={i + 1}
              team={r.team}
              attackScore={attackScore(r.attack)}
              defenceScore={defenceScore(r.defence)}
              confidence={r.confidence}
              fitted={r.fitted}
            />
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
