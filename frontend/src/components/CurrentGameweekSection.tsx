import type { CurrentGameweekResponse } from "../types";
import { CurrentGameweekCard } from "./CurrentGameweekCard";
import { EmptyState, RoundNavigator } from "../predictor-ui";
import { kickoffZones } from "../lib/kickoffTime";
import { NextFixtureHero } from "./NextFixtureHero";

interface Props {
  data: CurrentGameweekResponse;
  onSelect: (eventId: string) => void;
  /** A gameweek number, or undefined for "the current gameweek". */
  onNavigate: (gameweek: number | undefined) => void;
}

// A PRE-KICKOFF tally, and it says so: `RoundNavigator` renders this as
// "X/Y picks made before kickoff correct", which is exactly what it is. It is
// NOT the track record's headline — since the 2026-10-01 reversal that headline
// counts every recorded pick, whenever it was made (see
// `TrackRecordPanel.tsx`).
//
// The filter reads `made_before_kickoff`, which the backend DERIVES per read by
// comparing each pick's own `snapshotted_at` to its fixture's kickoff as UTC
// instants (`tracking/store.py::_made_before_kickoff`) — the same derivation the
// track record's `pre_kickoff.n_resolved_fixtures` and `/facts`'s
// `pick_timing` use. It used to read `!f.backfilled`, which is a different
// question with the same-looking answer: `backfilled` is a provenance flag
// ("this row came from the backfill job") whose writer takes it as an argument,
// so a pick captured by the five-minute tick AFTER its own kickoff is labelled
// live-captured. Two surfaces, two definitions of "before kickoff", one
// product.
//
// Picks recorded after their own kickoff are still listed on their cards (with
// the moment badge) and are still counted in the record; they are simply not in
// this particular figure. A payload that omits the field entirely — an older
// baked snapshot — contributes nothing here, which is the fail-closed
// direction: the label is withheld, never invented.
export function prekickoffTally(fixtures: { finished: boolean; hit: boolean | null; made_before_kickoff?: boolean }[]) {
  const finished = fixtures.filter((f) => f.finished);
  const counted = finished.filter((f) => f.made_before_kickoff);
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

  const zones = kickoffZones(data.fixtures.map((f) => f.commence_time));

  return (
    <div className="mb-8">
      <RoundNavigator
        label={data.gameweek !== null ? `Gameweek ${data.gameweek}` : "No gameweek data yet"}
        unit="gameweek"
        canPrev={canGoPrev}
        canNext={canGoNext}
        onPrev={() => data.gameweek !== null && onNavigate(data.gameweek - 1)}
        onNext={() => data.gameweek !== null && onNavigate(data.gameweek + 1)}
        onJumpToCurrent={data.is_current ? undefined : () => onNavigate(undefined)}
        record={tally}
      />
      {zones && <p className="-mt-2 mb-4 text-xs text-pl-text-dim">Kickoff times in {zones}</p>}

      {data.fixtures.length === 0 ? (
        <EmptyState
          message="No fixtures found for this gameweek."
          action={canGoNext && data.gameweek !== null ? { label: "Go to next gameweek", onClick: () => onNavigate(data.gameweek! + 1) } : undefined}
        />
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
