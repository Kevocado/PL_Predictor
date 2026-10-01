import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

// The panel BEFORE the 2026-10-01 reversal, kept as the fixture for the tests
// that pin what did not change.
//
// Under the old rule the headline was the pre-kickoff subset, so a record of 3
// rebuilt picks rendered as "—" with a line promising a figure the panel never
// drew. Two things here must survive the reversal, and both are asserted below:
//
//   * a payload with NOTHING at all (no picks, rebuilt or not) still takes the
//     short empty-state branch rather than rendering an empty panel;
//   * no NaN, which is the crash a null rate multiplied by a count invites.
//
// The substantive new behaviour is in `TrackRecordPanel.countedPicks.test.tsx`.

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
      n_resolved_fixtures: 0, pct_correct_overall: null,
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
    // And it must NOT claim a pre-kickoff figure it does not have.
    expect(screen.queryByText(/made before kickoff/i)).toBeNull();
  });

  it("renders no NaN when the record has resolved nothing", () => {
    render(<TrackRecordPanel data={nothingRecordedYet} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });

  it("does not crash when by_market is absent from the payload", () => {
    // A public_snapshot.json baked before that field existed can be live for a
    // few minutes after a deploy — the frontend ships instantly, the data it
    // reads is regenerated on its own schedule. This crashed the whole Data Hub
    // page the last time it was unguarded.
    const { by_market: _drop, ...summary } = nothingRecordedYet.summary;
    render(<TrackRecordPanel data={{ ...nothingRecordedYet, summary }} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });
});
