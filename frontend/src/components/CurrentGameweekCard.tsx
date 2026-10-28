import type { CurrentGameweekFixture } from "../types";
import { kickoffParts } from "../lib/kickoffTime";
import { TeamBadge } from "./TeamBadge";
import { ProbabilityBar } from "./ProbabilityBar";
import { FinishedFixtureCard } from "./FinishedFixtureCard";
import { pickCall } from "../lib/pick";

function formatKickoff(iso: string): { date: string; time: string } {
  const { day, time } = kickoffParts(iso);
  return { date: day, time };
}

export function CurrentGameweekCard({ fixture, onClick }: { fixture: CurrentGameweekFixture; onClick: () => void }) {
  const { date, time } = formatKickoff(fixture.commence_time);

  if (!fixture.finished) {
    const call = pickCall(fixture.predicted_home_win, fixture.predicted_draw, fixture.predicted_away_win, fixture.team_home, fixture.team_away);
    const value = fixture.value_bet_flags.length > 0;
    const likely = fixture.predicted_scoreline?.replace("-", "–");
    return (
      <div
        onClick={onClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          // Enter and Space, because this is announced as a button and must
          // behave like one. Space's default action is to scroll the page, so
          // preventDefault stops the keypress that opens the fixture from
          // also jumping the reader down the page.
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          onClick();
        }}
        className="pr-notch flex cursor-pointer flex-col gap-3 rounded-pr border border-pr-rule bg-pr-panel p-4 transition hover:border-pr-text-faint"
      >
        <div className="flex items-center justify-between gap-2 text-xs font-medium uppercase tracking-wide text-pr-text-faint">
          <span>
            {date} &middot; {time}
          </span>
          <span className="normal-case tracking-normal">{fixture.has_live_odds ? "Upcoming" : "Upcoming · model only"}</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
            <TeamBadge team={fixture.team_home} />
            <span className="text-xs font-semibold leading-tight text-pr-text">{fixture.team_home}</span>
          </div>
          <span className="font-display text-sm uppercase tracking-wide text-pr-text-faint">v</span>
          <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
            <TeamBadge team={fixture.team_away} />
            <span className="text-xs font-semibold leading-tight text-pr-text">{fixture.team_away}</span>
          </div>
        </div>
        <p className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 text-sm">
          <span className="font-semibold text-pr-text">
            {call.tossUp ? call.text : `Pick: ${call.text}`}
            {value && (
              <span
                title={`Model probability beats the live market's implied odds on: ${fixture.value_bet_flags.join(", ")}`}
                className="ml-2 rounded-pr bg-pr-accent px-1.5 py-0.5 text-xs font-semibold uppercase tracking-wide text-pr-accent-ink"
              >
                Value
              </span>
            )}
          </span>
          {likely && <span className="text-pr-text-dim">Most likely score {likely}</span>}
        </p>
        <ProbabilityBar
          home={fixture.predicted_home_win}
          draw={fixture.predicted_draw}
          away={fixture.predicted_away_win}
          pick={call.tossUp ? null : call.side}
        />
      </div>
    );
  }

  return <FinishedFixtureCard {...fixture} onClick={onClick} />;
}
