import { TeamBadge } from "./TeamBadge";
import type { FixturePlayerEvent } from "../types";

function formatKickoff(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }),
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
  };
}

interface Props {
  commence_time: string;
  team_home: string;
  team_away: string;
  actual_goals_home: number | null;
  actual_goals_away: number | null;
  predicted_scoreline: string | null;
  predicted_home_win: number;
  predicted_draw: number;
  predicted_away_win: number;
  hit: boolean | null;
  backfilled: boolean;
  home_player_events?: FixturePlayerEvent[];
  away_player_events?: FixturePlayerEvent[];
  player_events_pending?: boolean;
  onClick: () => void;
}

function playerSummary(events: FixturePlayerEvent[], key: "goals" | "assists", label: string) {
  const contributors = events.filter((event) => event[key] > 0);
  if (!contributors.length) return null;
  return `${label}: ${contributors.map((event) => `${event.name}${event[key] > 1 ? ` (${event[key]})` : ""}`).join(", ")}`;
}

// Shared "already-played fixture" card — same visual language wherever a
// finished match with an honest pre-match prediction is shown (the
// Fixtures page's current-gameweek view, the Track Record tab's
// per-gameweek results). Clicking opens the same full fixture-detail
// modal both places already use.
export function FinishedFixtureCard({
  commence_time,
  team_home,
  team_away,
  actual_goals_home,
  actual_goals_away,
  predicted_scoreline,
  predicted_home_win,
  predicted_draw,
  predicted_away_win,
  hit,
  backfilled,
  home_player_events = [],
  away_player_events = [],
  player_events_pending = false,
  onClick,
}: Props) {
  const { date, time } = formatKickoff(commence_time);

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
      className="clip-corner flex cursor-pointer flex-col gap-3 rounded-xl border border-win/30 bg-pl-850/70 p-4 transition hover:border-pl-pink/40"
    >
      <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-pl-text-faint">
        <span>
          {date} &middot; {time}
        </span>
        <div className="flex items-center gap-1.5">
          {backfilled && (
            <span
              title="This match had already finished before the app started tracking it live — the prediction shown is what the model would have said, computed the same way as any live prediction."
              className="rounded border border-pl-border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-pl-text-faint"
            >
              Backfilled
            </span>
          )}
          <span className={`font-semibold ${hit ? "text-win" : "text-loss"}`}>{hit ? "Called it ✓" : "Missed ✗"}</span>
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={team_home} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{team_home}</span>
        </div>
        <div className="flex flex-col items-center gap-0.5 px-1">
          <span className="text-[10px] font-medium uppercase text-pl-text-faint">Final</span>
          <span className="rounded-lg bg-pl-700/50 px-2 py-1 font-mono text-base font-bold text-pl-text">
            {actual_goals_home}-{actual_goals_away}
          </span>
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={team_away} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{team_away}</span>
        </div>
      </div>
      {(home_player_events.length > 0 || away_player_events.length > 0) && (
        <div className="grid grid-cols-2 gap-3 border-t border-pl-border/70 pt-2 text-[10px] leading-relaxed text-pl-text-dim">
          <div>{playerSummary(home_player_events, "goals", "Goals") && <p>{playerSummary(home_player_events, "goals", "Goals")}</p>}{playerSummary(home_player_events, "assists", "Assists") && <p>{playerSummary(home_player_events, "assists", "Assists")}</p>}</div>
          <div className="text-right">{playerSummary(away_player_events, "goals", "Goals") && <p>{playerSummary(away_player_events, "goals", "Goals")}</p>}{playerSummary(away_player_events, "assists", "Assists") && <p>{playerSummary(away_player_events, "assists", "Assists")}</p>}</div>
        </div>
      )}
      {player_events_pending && (
        <p className="border-t border-pl-border/70 pt-2 text-center text-[10px] text-pl-text-faint">Refreshing official scorers and assists…</p>
      )}
      <div className="flex items-center justify-between border-t border-pl-border/70 pt-2.5 text-[11px] text-pl-text-dim">
        <span>
          We predicted <span className="font-mono font-semibold text-pl-text">{predicted_scoreline ?? "?"}</span>
        </span>
        <span>
          {(predicted_home_win * 100).toFixed(0)}% / {(predicted_draw * 100).toFixed(0)}% / {(predicted_away_win * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
