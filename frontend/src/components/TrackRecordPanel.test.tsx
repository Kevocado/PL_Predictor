// The PL track record panel, rebuilt on the shared kit.
//
// PL's counting rule already matches PRODUCT.md: the headline is every counted
// pick, and `pre_kickoff` is the subset made before kickoff with its own n.
// The kit now renders that as a headline with a Wilson range instead of PL's
// old `(2/3)` StatCard text.
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

const nothingRecordedYet: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 0, n_rebuilt_fixtures: 0, pct_correct_overall: null, current_gameweek: null,
    pct_correct_current_gameweek: null, n_fixtures_current_gameweek: 0,
    gameweek_trend: [],
    by_market: {
      exact_score: { pct_correct: null, n_resolved: 0 },
      match_result: { pct_correct: null, n_resolved: 0 },
      over_under_2_5: { pct_correct: null, n_resolved: 0 },
      btts: { pct_correct: null, n_resolved: 0 },
    },
    pre_kickoff: {
      n_resolved_fixtures: 0, pct_correct_overall: null, n_rebuilt_fixtures: 0,
      by_market: {
        exact_score: { pct_correct: null, n_resolved: 0 },
        match_result: { pct_correct: null, n_resolved: 0 },
        over_under_2_5: { pct_correct: null, n_resolved: 0 },
        btts: { pct_correct: null, n_resolved: 0 },
      },
    },
    per_pick: [],
  },
  biggest_upsets: [],
  gameweeks: [],
};

describe("TrackRecordPanel empty states", () => {
  it("says nothing is recorded yet when there are no picks at all", () => {
    render(<TrackRecordPanel data={nothingRecordedYet} />);
    expect(screen.getByText(/No scored predictions yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/made before kickoff/i)).toBeNull();
  });

  it("renders no NaN when the record has resolved nothing", () => {
    render(<TrackRecordPanel data={nothingRecordedYet} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });

  it("does not crash when by_market is absent from the payload", () => {
    const { by_market: _drop, ...summary } = nothingRecordedYet.summary;
    render(<TrackRecordPanel data={{ ...nothingRecordedYet, summary }} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });
});
