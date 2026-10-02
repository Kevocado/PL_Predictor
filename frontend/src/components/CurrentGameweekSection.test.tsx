import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { prekickoffTally } from "./CurrentGameweekSection";

// The tests below import Testing Library lazily, so unmount explicitly.
afterEach(cleanup);

// `madeBeforeKickoff` is the derived flag and the only thing the tally reads.
// `backfilled` is carried too, and passed explicitly wherever it matters, so a
// test can show the two disagreeing -- which is the whole point.
// The annotation is what lets `backfilled` ride along: `prekickoffTally`'s
// parameter type is deliberately the narrowest one it reads, and a bare object
// literal assigned to it would otherwise be an excess-property error rather than
// a test that can show the two fields disagreeing.
type TallyInput = Parameters<typeof prekickoffTally>[number][number];
const f = (
  finished: boolean,
  hit: boolean | null,
  madeBeforeKickoff: boolean,
  backfilled = false,
): TallyInput & { backfilled: boolean } => ({
  finished,
  hit,
  backfilled,
  made_before_kickoff: madeBeforeKickoff,
});

describe("prekickoffTally", () => {
  it("counts only finished picks made before kickoff", () => {
    expect(prekickoffTally([
      f(true, true, false, true),  // made late
      f(true, false, true),       // made in time, missed
      f(true, true, true),        // made in time, hit
      f(false, null, false),      // not finished
    ])).toEqual({ hits: 1, settled: 2, rebuilt: 1 });
  });
  it("reports zero settled when every finished pick was rebuilt", () => {
    expect(prekickoffTally([f(true, true, false, true), f(true, false, false, true)]))
      .toEqual({ hits: 0, settled: 0, rebuilt: 2 });
  });

  // These two are the defect. `backfilled` is provenance -- which job wrote the
  // row -- and the five-minute tracking tick writes `false` whenever it runs,
  // including an hour after kickoff. Reading it made this tally claim picks the
  // track record's own `pre_kickoff` figure cannot find.
  it("reads the derived flag, not backfilled, when the two disagree", () => {
    const tally = prekickoffTally([
      f(true, true, true, true),   // made in time, yet flagged backfilled by the backfill job
      f(true, true, false, false), // written late by the tick, hence unflagged
    ]);
    expect(tally).toEqual({ hits: 1, settled: 1, rebuilt: 1 });
  });

  it("fails closed when a payload predates the derived field", () => {
    // A snapshot baked before `made_before_kickoff` existed. Withholding the
    // label is right; inventing it would print a pre-kickoff figure nobody can
    // place in time.
    const tally = prekickoffTally([
      { ...f(true, true, false), made_before_kickoff: undefined },
      { ...f(true, false, false), made_before_kickoff: undefined },
    ]);
    expect(tally).toEqual({ hits: 0, settled: 0, rebuilt: 2 });
  });

  it("agrees with the track record's pre-kickoff count for the same gameweek", () => {
    // The shipped `public_snapshot.json` shape: every finished fixture listed,
    // and the summary's `pre_kickoff.n_resolved_fixtures` counting exactly the
    // ones whose own timestamps put them in time.
    const fixtures = [
      f(true, true, true), f(true, false, true), f(true, true, false), f(false, null, false),
    ];
    const preKickoff = { n_resolved_fixtures: 2, pct_correct_overall: 0.5 };
    expect(prekickoffTally(fixtures).settled).toBe(preKickoff.n_resolved_fixtures);
    expect(prekickoffTally(fixtures).hits).toBe(Math.round(preKickoff.pct_correct_overall! * 2));
  });
});

describe("gameweek headline copy", () => {
  it("does not promise that rebuilt-only gameweeks will settle later", async () => {
    const { render, screen } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const fixture = {
      event_id: "e1", team_home: "Tottenham", team_away: "Aston Villa", commence_time: "2026-09-19T10:30:00Z",
      finished: true, actual_goals_home: 2, actual_goals_away: 3, predicted_home_win: 0.36, predicted_draw: 0.26,
      predicted_away_win: 0.38, predicted_scoreline: "1-1", draw_signal: false, hit: true, backfilled: true,
      made_before_kickoff: false,
      has_live_odds: false, value_bet_flags: [], home_player_events: [], away_player_events: [], player_events_pending: false,
    };
    const data = { gameweek: 5, is_current: true, min_gameweek: 1, max_gameweek: 38, fixtures: [fixture] };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={() => {}} />);
    expect(screen.getByText("No picks made before kickoff this gameweek")).toBeInTheDocument();
  });
});

describe("gameweek navigator", () => {
  const fixture = (id: string, commence_time: string, over: Record<string, unknown> = {}) => ({
    event_id: id, team_home: "Tottenham", team_away: "Aston Villa", commence_time,
    finished: false, actual_goals_home: null, actual_goals_away: null, predicted_home_win: 0.36, predicted_draw: 0.26,
    predicted_away_win: 0.38, predicted_scoreline: "1-1", draw_signal: false, hit: null, backfilled: false,
    made_before_kickoff: false,
    has_live_odds: true, value_bet_flags: [], home_player_events: [], away_player_events: [], player_events_pending: false, ...over,
  });

  it("heads the gameweek with its record and says which zone kickoff times are in", async () => {
    const { render, screen } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const data = {
      gameweek: 6, is_current: true, min_gameweek: 1, max_gameweek: 38,
      fixtures: [
        // Finished and made in time -- the derived flag says so, and it is the
        // only thing that does. `backfilled` stays false here to match a real
        // tracking-tick capture.
        fixture("a", "2026-09-26T11:30:00Z", { finished: true, actual_goals_home: 2, actual_goals_away: 0, hit: true, made_before_kickoff: true }),
        fixture("b", "2026-09-27T14:00:00Z"),
      ],
    };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={() => {}} />);
    expect(screen.getByRole("heading", { name: "Gameweek 6" })).toBeInTheDocument();
    expect(screen.getByText("1/1 picks made before kickoff correct")).toBeInTheDocument();
    expect(screen.getByText("Kickoff times in CDT")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Jump to current gameweek" })).not.toBeInTheDocument();
  });

  it("offers a jump back to the current gameweek instead of a 'not current' pill", async () => {
    const { render, screen, fireEvent } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const onNavigate = vi.fn();
    const data = { gameweek: 3, is_current: false, min_gameweek: 1, max_gameweek: 38, fixtures: [fixture("a", "2026-08-30T14:00:00Z")] };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={onNavigate} />);
    expect(screen.queryByText(/not current/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Jump to current gameweek" }));
    expect(onNavigate).toHaveBeenCalledWith(undefined);
    fireEvent.click(screen.getByRole("button", { name: "Next gameweek" }));
    expect(onNavigate).toHaveBeenLastCalledWith(4);
  });
});
