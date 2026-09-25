import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

const rebuiltFixture = {
  event_id: "e1", team_home: "Tottenham", team_away: "Aston Villa", commence_time: "2026-09-19T10:30:00Z",
  predicted_scoreline: "1-1", actual_goals_home: 2, actual_goals_away: 3,
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  actual_outcome: "away_win" as const, hit: true, backfilled: true,
};

const onlyRebuilt: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 0, n_rebuilt_fixtures: 3, pct_correct_overall: null, current_gameweek: null,
    pct_correct_current_gameweek: null, n_fixtures_current_gameweek: 0, gameweek_trend: [],
  },
  biggest_upsets: [],
  gameweeks: [{ gameweek: 5, pct_correct: null, n_fixtures: 0, n_rebuilt: 3, fixtures: [rebuiltFixture] }],
};

describe("TrackRecordPanel", () => {
  it("says rebuilt picks are shown but not counted, and still lists them", () => {
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(screen.getByText("3 picks rebuilt after kickoff are shown but not counted.")).toBeInTheDocument();
    expect(screen.getByText("Rebuilt after kickoff")).toBeInTheDocument();
  });
  it("shows a dash for the overall rate and no NaN in gameweek headers", () => {
    render(<TrackRecordPanel data={onlyRebuilt} />);
    expect(screen.getByText("No pre-kickoff picks")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN/);
  });
});
