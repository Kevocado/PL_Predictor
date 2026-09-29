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

// Rebuilt picks count in every rate under the score-everything rule:
// 2 hits of 3 resolved, all 3 rebuilt.
const onlyRebuilt: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 3, n_rebuilt_fixtures: 3, pct_correct_overall: 2 / 3, current_gameweek: 5,
    pct_correct_current_gameweek: 2 / 3, n_fixtures_current_gameweek: 3,
    gameweek_trend: [{ gameweek: 5, pct_correct: 2 / 3, n_fixtures: 3 }],
    by_market: {
      exact_score: { pct_correct: 0, n_resolved: 3 },
      match_result: { pct_correct: 2 / 3, n_resolved: 3 },
      over_under_2_5: { pct_correct: 1 / 3, n_resolved: 3 },
      btts: { pct_correct: 1, n_resolved: 3 },
    },
  },
  biggest_upsets: [],
  gameweeks: [{
    gameweek: 5, pct_correct: 2 / 3, n_fixtures: 3, n_rebuilt: 3,
    pct_correct_by_market: { exact_score: 0, match_result: 2 / 3, over_under_2_5: 1 / 3, btts: 1 },
    fixtures: [rebuiltFixture("e1", true), rebuiltFixture("e2", true), rebuiltFixture("e3", false)],
  }],
};

describe("TrackRecordPanel", () => {
  it("says rebuilt picks are included in the score, and still lists them", () => {
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(screen.getByText("3 picks rebuilt after kickoff are included in the score.")).toBeInTheDocument();
    expect(screen.getAllByText("Rebuilt after kickoff").length).toBeGreaterThan(0);
  });
  it("scores rebuilt picks in the headline rate and gameweek headers with no NaN", () => {
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(screen.queryByText("No pre-kickoff picks")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN/);
    const overall = screen.getByText(/Correct overall/i).parentElement;
    expect(overall).toHaveTextContent("67%");
    expect(overall).toHaveTextContent("2/3");
  });
});
