import type { CurrentGameweekFixture } from "../types";
import { TeamBadge } from "./TeamBadge";
import { ProbabilityBar } from "./ProbabilityBar";

function formatKickoff(iso: string): { day: string; date: string; time: string } {
  const d = new Date(iso);
  return {
    day: d.toLocaleDateString(undefined, { weekday: "long" }),
    date: d.toLocaleDateString(undefined, { day: "numeric", month: "long" }),
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
  };
}

// The next fixture yet to kick off gets one clearly bigger treatment per
// gameweek — the same "one hero" energy the FPL pitch view already has,
// which the Fixtures grid (identical card weight for all ten matches)
// didn't. Everything else in the gameweek stays in the regular grid below.
export function NextFixtureHero({ fixture, onClick }: { fixture: CurrentGameweekFixture; onClick: () => void }) {
  const { day, date, time } = formatKickoff(fixture.commence_time);

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
      className="clip-corner-lg relative mb-6 cursor-pointer overflow-hidden rounded-2xl border border-pl-pink/30 bg-gradient-to-br from-pl-800 via-pl-850 to-pl-900 p-6 transition hover:border-pl-pink/60 sm:p-8"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-medium uppercase tracking-wide text-pl-pink">
        <span>Next up</span>
        <div className="flex items-center gap-1.5 normal-case text-pl-text-dim">
          {fixture.value_bet_flags.length > 0 && (
            <span
              title={`Model probability beats the live market's implied odds on: ${fixture.value_bet_flags.join(", ")}`}
              className="rounded bg-pl-pink/20 px-1.5 py-0.5 text-xs font-semibold uppercase text-pl-pink"
            >
              Value bet
            </span>
          )}
          {!fixture.has_live_odds && <span className="text-pl-text-faint">Model only</span>}
        </div>
      </div>

      <div className="mt-5 grid grid-cols-[1fr_auto_1fr] items-center gap-4 sm:gap-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <TeamBadge team={fixture.team_home} size="lg" />
          <span className="font-display text-lg font-semibold tracking-wide text-pl-text sm:text-xl">{fixture.team_home}</span>
        </div>

        <div className="flex flex-col items-center gap-1 px-2">
          <span className="font-display text-sm font-medium tracking-wide text-pl-text-faint">{day}</span>
          <span className="font-display text-2xl font-semibold tracking-wide text-pl-text sm:text-3xl">{time}</span>
          <span className="text-xs text-pl-text-faint">{date}</span>
          <span className="mt-2 rounded-lg bg-pl-950/50 px-3 py-1 font-display text-lg font-semibold tracking-wide text-pl-text-dim">
            {fixture.predicted_scoreline?.replace("-", "–") ?? "?"}
          </span>
        </div>

        <div className="flex flex-col items-center gap-3 text-center">
          <TeamBadge team={fixture.team_away} size="lg" />
          <span className="font-display text-lg font-semibold tracking-wide text-pl-text sm:text-xl">{fixture.team_away}</span>
        </div>
      </div>

      <div className="mx-auto mt-6 max-w-md">
        <ProbabilityBar
          home={fixture.predicted_home_win}
          draw={fixture.predicted_draw}
          away={fixture.predicted_away_win}
          homeLabel={fixture.team_home}
          awayLabel={fixture.team_away}
        />
      </div>
    </div>
  );
}
