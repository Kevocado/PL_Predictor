import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

const rebuiltFixture = (event_id: string, hit: boolean) => ({
  event_id, team_home: "Tottenham", team_away: "Aston Villa", commence_time: "2026-09-19T10:30:00Z",
  predicted_scoreline: "1-1", actual_goals_home: 2, actual_goals_away: 3,
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  actual_outcome: "away_win" as const, hit, backfilled: true,
});

// The shape the backend sends under B8 when EVERY stored pick was rebuilt:
// zero pre-kickoff picks, so the headline has nothing to report, and the
// rebuilt picks live in `all_picks` / the per-fixture list rather than in the
// headline. This fixture used to be the old one (n_resolved 3, headline 2/3),
// which is precisely the claim the change removes.
const onlyRebuilt: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 0, n_rebuilt_fixtures: 3, pct_correct_overall: null, current_gameweek: 5,
    pct_correct_current_gameweek: null, n_fixtures_current_gameweek: 0,
    gameweek_trend: [],
    by_market: {
      exact_score: { pct_correct: null, n_resolved: 0 },
      match_result: { pct_correct: null, n_resolved: 0 },
      over_under_2_5: { pct_correct: null, n_resolved: 0 },
      btts: { pct_correct: null, n_resolved: 0 },
    },
    all_picks: {
      n_resolved: 3, pct_correct: 2 / 3,
      by_market: {
        exact_score: { pct_correct: 0, n_resolved: 3 },
        match_result: { pct_correct: 2 / 3, n_resolved: 3 },
        over_under_2_5: { pct_correct: 1 / 3, n_resolved: 3 },
        btts: { pct_correct: 1, n_resolved: 3 },
      },
    },
    per_pick: [
      { event_id: "e1", team_home: "A", team_away: "B", gameweek: 5, hit: true, rebuilt: true },
      { event_id: "e2", team_home: "C", team_away: "D", gameweek: 5, hit: true, rebuilt: true },
      { event_id: "e3", team_home: "E", team_away: "F", gameweek: 5, hit: false, rebuilt: true },
    ],
  },
  biggest_upsets: [],
  gameweeks: [{
    gameweek: 5, pct_correct: 2 / 3, n_fixtures: 3, n_rebuilt: 3,
    pct_correct_by_market: { exact_score: 0, match_result: 2 / 3, over_under_2_5: 1 / 3, btts: 1 },
    fixtures: [rebuiltFixture("e1", true), rebuiltFixture("e2", true), rebuiltFixture("e3", false)],
  }],
};

describe("TrackRecordPanel", () => {
  it("does NOT count rebuilt picks toward the headline, and says where they went", () => {
    // This test used to assert the opposite — "included in the score" — which
    // is the claim B8 rules out. The rewritten assertion is the same fixture
    // (3 picks, all rebuilt) with the ruling's answer instead of the old one.
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(
      screen.getByText(/not in the headline above/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/included in the score/i)).toBeNull();
    // Still listed, so scoping the headline does not hide the picks.
    expect(screen.getAllByText("Rebuilt after kickoff").length).toBeGreaterThan(0);
  });

  it("shows no headline rate when every stored pick was rebuilt", () => {
    // 0 out of 0 is an absence, not 0%. Rendering "67% (2/3)" here — which is
    // what this test asserted before — is the look-forward bias the whole
    // change exists to remove, and it is what the live site was showing.
    render(<TrackRecordPanel data={onlyRebuilt} />);
    const overall = screen.getByText(/Correct overall/i).parentElement;
    expect(overall).toHaveTextContent("—");
    expect(overall).not.toHaveTextContent("67%");
    expect(overall).not.toHaveTextContent("2/3");
  });

  it("says plainly that there is no honest score yet, rather than 'no scored predictions'", () => {
    // The old empty-state copy claimed nothing had been scored, which is false
    // here: 3 picks exist. They were all made too late to count.
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(screen.getByText(/no honest score to show/i)).toBeInTheDocument();
    expect(screen.queryByText(/No scored predictions yet/i)).toBeNull();
  });

  it("renders no NaN anywhere in the no-pre-kickoff-picks state", () => {
    // Kept as its own test because it is the crash this state invites: a null
    // rate multiplied by a count, or a percentage of nothing.
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });
});
