import type { CurrentGameweekFixture } from "../types";
import { TeamBadge } from "./TeamBadge";
import { ProbabilityBar } from "./ProbabilityBar";

function formatKickoff(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }),
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
  };
}

export function CurrentGameweekCard({ fixture, onClick }: { fixture: CurrentGameweekFixture; onClick: () => void }) {
  const { date, time } = formatKickoff(fixture.commence_time);

  if (!fixture.finished) {
    return (
      <div
        onClick={onClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && onClick()}
        className={`clip-corner flex cursor-pointer flex-col gap-3 rounded-xl border bg-pl-850/70 p-4 transition hover:border-pl-pink/40 ${
          fixture.value_bet_flags.length > 0 ? "border-pl-pink/40 ring-1 ring-pl-pink/20" : "border-pl-border"
        }`}
      >
        <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-pl-text-faint">
          <span>
            {date} &middot; {time}
          </span>
          <div className="flex items-center gap-1.5">
            {fixture.value_bet_flags.length > 0 && (
              <span
                title={`Model probability beats the live market's implied odds on: ${fixture.value_bet_flags.join(", ")}`}
                className="rounded bg-pl-pink/20 px-1.5 py-0.5 font-semibold text-pl-pink"
              >
                Value bet
              </span>
            )}
            {!fixture.has_live_odds && (
              <span title="No live bookmaker line for this fixture yet — model-only prediction." className="text-pl-text-faint">
                Model only
              </span>
            )}
            <span className="rounded bg-pl-700/60 px-1.5 py-0.5 text-pl-text-dim">Upcoming</span>
          </div>
        </div>
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
            <TeamBadge team={fixture.team_home} />
            <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_home}</span>
          </div>
          <div className="flex flex-col items-center gap-0.5 px-1">
            <span className="text-[10px] font-medium uppercase text-pl-text-faint">Likely</span>
            <span className="rounded-lg bg-pl-700/50 px-2 py-1 font-mono text-base font-bold text-pl-text">
              {fixture.predicted_scoreline ?? "?"}
            </span>
          </div>
          <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
            <TeamBadge team={fixture.team_away} />
            <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_away}</span>
          </div>
        </div>
        <ProbabilityBar home={fixture.predicted_home_win} draw={fixture.predicted_draw} away={fixture.predicted_away_win} />
      </div>
    );
  }

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
          {fixture.backfilled && (
            <span
              title="This match had already finished before the app started tracking it live — the prediction shown is what the model would have said, computed the same way as any live prediction."
              className="rounded border border-pl-border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-pl-text-faint"
            >
              Backfilled
            </span>
          )}
          <span className={`font-semibold ${fixture.hit ? "text-win" : "text-loss"}`}>{fixture.hit ? "Called it ✓" : "Missed ✗"}</span>
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={fixture.team_home} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_home}</span>
        </div>
        <div className="flex flex-col items-center gap-0.5 px-1">
          <span className="text-[10px] font-medium uppercase text-pl-text-faint">Final</span>
          <span className="rounded-lg bg-pl-700/50 px-2 py-1 font-mono text-base font-bold text-pl-text">
            {fixture.actual_goals_home}-{fixture.actual_goals_away}
          </span>
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={fixture.team_away} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_away}</span>
        </div>
      </div>
      <div className="flex items-center justify-between border-t border-pl-border/70 pt-2.5 text-[11px] text-pl-text-dim">
        <span>
          We predicted <span className="font-mono font-semibold text-pl-text">{fixture.predicted_scoreline ?? "?"}</span>
        </span>
        <span>
          {(fixture.predicted_home_win * 100).toFixed(0)}% / {(fixture.predicted_draw * 100).toFixed(0)}% /{" "}
          {(fixture.predicted_away_win * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
