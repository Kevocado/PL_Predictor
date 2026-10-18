// The PL track record panel after the 2026-10-01 reversal, on the shared kit.
//
// B8's two figures swapped roles. The headline is now every COUNTED pick,
// whenever it was made; the figure beside it is the subset made before kickoff,
// with its own n.
import { describe, expect, it } from "vitest";
import { render, within } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

// 3 counted picks, all recorded after their own kickoff. `pre_kickoff` is empty
// because no pick's own timestamps prove it was made in time.
const allLate: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 3, n_rebuilt_fixtures: 3, pct_correct_overall: 2 / 3,
    current_gameweek: 5,
    pct_correct_current_gameweek: 2 / 3, n_fixtures_current_gameweek: 3,
    gameweek_trend: [],
    by_market: {
      exact_score: { pct_correct: 0, n_resolved: 3 },
      match_result: { pct_correct: 2 / 3, n_resolved: 3 },
      over_under_2_5: { pct_correct: 1 / 3, n_resolved: 3 },
      btts: { pct_correct: 1, n_resolved: 3 },
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
    per_pick: [
      { event_id: "e1", team_home: "A", team_away: "B", gameweek: 5, hit: true, made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e2", team_home: "C", team_away: "D", gameweek: 5, hit: true, made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e3", team_home: "E", team_away: "F", gameweek: 5, hit: false, made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
    ],
  },
  biggest_upsets: [],
  gameweeks: [],
};

// 3 counted picks, one of them made in time. The two figures genuinely differ.
const mixed: TrackRecordResponse = {
  ...allLate,
  summary: {
    ...allLate.summary,
    n_resolved_fixtures: 3, n_rebuilt_fixtures: 2, pct_correct_overall: 1 / 3,
    pct_correct_current_gameweek: 1 / 3, n_fixtures_current_gameweek: 3,
    by_market: {
      exact_score: { pct_correct: 0, n_resolved: 3 },
      match_result: { pct_correct: 1 / 3, n_resolved: 3 },
      over_under_2_5: { pct_correct: 0, n_resolved: 3 },
      btts: { pct_correct: 1 / 3, n_resolved: 3 },
    },
    pre_kickoff: {
      n_resolved_fixtures: 1, pct_correct_overall: 0, n_rebuilt_fixtures: 0,
      by_market: {
        exact_score: { pct_correct: null, n_resolved: 1 },
        match_result: { pct_correct: 0, n_resolved: 1 },
        over_under_2_5: { pct_correct: null, n_resolved: 1 },
        btts: { pct_correct: null, n_resolved: 1 },
      },
    },
    per_pick: [
      { event_id: "e1", team_home: "A", team_away: "B", gameweek: 5, hit: false, made_before_kickoff: true, snapshotted_at: "2026-09-19T08:00:00Z", rebuilt: false },
      { event_id: "e2", team_home: "C", team_away: "D", gameweek: 5, hit: false, made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e3", team_home: "E", team_away: "F", gameweek: 5, hit: true, made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
    ],
  },
};

function tally(container: HTMLElement, market: string): HTMLElement {
  return within(container).getByTestId(`tally-${market}`);
}

describe("TrackRecordPanel counts every recorded pick", () => {
  it("puts the whole record in the headline, with a range, even when every pick was made late", () => {
    const { container } = render(<TrackRecordPanel data={allLate} />);
    // match_result: 2 of 3, 67%, with a Wilson range beside it.
    expect(tally(container, "match_result")).toHaveTextContent("2 of 3");
    expect(tally(container, "match_result")).toHaveTextContent("67%");
    expect(tally(container, "match_result")).toHaveTextContent("Could plausibly be");
    // The headline is 2/3, not "—" — the old panel rendered "—" here while 3
    // graded picks sat in the same payload.
    expect(tally(container, "match_result")).not.toHaveTextContent("—");
  });

  it("shows the made-before-kickoff subset beside the headline, with its own n", () => {
    const { container } = render(<TrackRecordPanel data={mixed} />);
    // Headline: 1 of 3, 33%. Beside it: 0 of 1 made before kickoff.
    expect(tally(container, "match_result")).toHaveTextContent("1 of 3");
    expect(tally(container, "match_result")).toHaveTextContent("33%");
    expect(tally(container, "match_result")).toHaveTextContent("0 of 1 made before kickoff");
    // And it is a DIFFERENT number from the headline, which is the reason both
    // are published.
    expect(tally(container, "match_result")).not.toHaveTextContent("2 of 3");
  });

  it("reads as an absence, not 0%, when nothing was picked in time", () => {
    const { container } = render(<TrackRecordPanel data={allLate} />);
    // Nothing made before kickoff: no "0 of 0" beside the headline.
    expect(tally(container, "match_result")).not.toHaveTextContent("made before kickoff");
    expect(tally(container, "match_result")).not.toHaveTextContent("0%");
  });

  it("never tells a reader a counted pick was held out of the record", () => {
    render(<TrackRecordPanel data={allLate} />);
    expect(document.body.textContent).not.toMatch(/not\s+counted/i);
    expect(document.body.textContent).not.toMatch(/not in the headline/i);
    expect(document.body.textContent).not.toMatch(/Rebuilt after/i);
  });

  it("discloses that its picks were made after kickoff", () => {
    const { container } = render(<TrackRecordPanel data={allLate} />);
    // All 3 were rebuilt after kickoff; the note states that plainly.
    expect(within(container).getByText(/3 counted picks in this record, of which 3 were made at or after their own kickoff/)).toBeInTheDocument();
    // And the pre-kickoff figure is an absence, not a second headline.
    expect(tally(container, "match_result")).not.toHaveTextContent("made before kickoff");
  });

  it("renders no NaN anywhere", () => {
    render(<TrackRecordPanel data={allLate} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });

  it("still renders a panel when the payload predates the pre_kickoff field", () => {
    const { pre_kickoff: _drop, ...summary } = allLate.summary;
    const { container } = render(<TrackRecordPanel data={{ ...allLate, summary }} />);
    expect(tally(container, "match_result")).toHaveTextContent("2 of 3");
  });
});
