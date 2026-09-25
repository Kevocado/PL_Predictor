import type { CurrentGameweekResponse } from "../types";
import { CurrentGameweekCard } from "./CurrentGameweekCard";
import { NextFixtureHero } from "./NextFixtureHero";

interface Props {
  data: CurrentGameweekResponse;
  onSelect: (eventId: string) => void;
  onNavigate: (gameweek: number) => void;
}

// Only picks captured before kickoff count; ones rebuilt after the match
// (backfilled) are listed on their cards but never scored here.
export function prekickoffTally(fixtures: { finished: boolean; hit: boolean | null; backfilled: boolean }[]) {
  const finished = fixtures.filter((f) => f.finished);
  const counted = finished.filter((f) => !f.backfilled);
  return { hits: counted.filter((f) => f.hit).length, settled: counted.length, rebuilt: finished.length - counted.length };
}

export function CurrentGameweekSection({ data, onSelect, onNavigate }: Props) {
  const tally = prekickoffTally(data.fixtures);

  // Only the current gameweek gets a hero -- browsing a past gameweek via
  // the arrows shouldn't promote one of its (already-played) fixtures, and
  // a future gameweek has no live odds/predictions worth spotlighting yet
  // over the others.
  const upcoming = data.is_current ? data.fixtures.filter((f) => !f.finished) : [];
  const nextFixture = upcoming.length > 0
    ? upcoming.reduce((soonest, f) => (new Date(f.commence_time) < new Date(soonest.commence_time) ? f : soonest))
    : null;
  const gridFixtures = nextFixture ? data.fixtures.filter((f) => f.event_id !== nextFixture.event_id) : data.fixtures;

  const canGoPrev = data.gameweek !== null && data.min_gameweek !== null && data.gameweek > data.min_gameweek;
  const canGoNext = data.gameweek !== null && data.max_gameweek !== null && data.gameweek < data.max_gameweek;

  return (
    <div className="mb-8">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            disabled={!canGoPrev}
            onClick={() => data.gameweek !== null && onNavigate(data.gameweek - 1)}
            aria-label="Previous gameweek"
            className="rounded-lg border border-pl-border bg-pl-850/70 px-2.5 py-1.5 text-sm text-pl-text-dim transition hover:text-pl-text disabled:cursor-not-allowed disabled:opacity-30"
          >
            ←
          </button>
          <h2 className="flex items-baseline gap-2 text-lg font-semibold text-pl-text">
            {data.gameweek !== null ? `Gameweek ${data.gameweek}` : "No gameweek data yet"}
            {!data.is_current && (
              <span className="rounded bg-pl-700/60 px-1.5 py-0.5 text-xs font-normal text-pl-text-dim">
                not current
              </span>
            )}
          </h2>
          <button
            disabled={!canGoNext}
            onClick={() => data.gameweek !== null && onNavigate(data.gameweek + 1)}
            aria-label="Next gameweek"
            className="rounded-lg border border-pl-border bg-pl-850/70 px-2.5 py-1.5 text-sm text-pl-text-dim transition hover:text-pl-text disabled:cursor-not-allowed disabled:opacity-30"
          >
            →
          </button>
        </div>
        <span className="text-xs font-normal text-pl-text-dim">
          {data.fixtures.length} fixture{data.fixtures.length === 1 ? "" : "s"}
          {tally.settled > 0 && ` · ${tally.hits}/${tally.settled} picks made before kickoff correct`}
          {tally.settled === 0 && tally.rebuilt > 0 && " · No pre-kickoff picks this gameweek"}
        </span>
      </div>

      {data.fixtures.length === 0 ? (
        <div className="py-16 text-center text-pl-text-faint">No fixtures found for this gameweek.</div>
      ) : (
        <>
          {nextFixture && <NextFixtureHero fixture={nextFixture} onClick={() => onSelect(nextFixture.event_id)} />}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {gridFixtures.map((f) => (
              <CurrentGameweekCard
                key={f.event_id}
                fixture={f}
                onClick={() => onSelect(f.event_id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
