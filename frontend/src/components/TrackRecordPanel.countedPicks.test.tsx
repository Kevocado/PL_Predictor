// The track-record panel after the 2026-10-01 reversal.
//
// B8's two figures swapped roles. The headline is now every COUNTED pick,
// whenever it was made; the figure beside it is the subset made before kickoff,
// with its own n. Before the change the headline was the pre-kickoff subset and
// the panel said of a rebuilt pick that it was "not in the headline" — which is
// now false, because it IS in the headline.
//
// The fixture below is PL's shipped state translated to a test: 3 picks, every
// one of them made after its own kickoff. Under the old rule that rendered as
// "—" beside a line promising a figure the panel never rendered. Under the new
// rule the headline is 2/3 and the secondary figure is an absence, clearly
// labelled.
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import type { TrackRecordResponse } from "../types";

const fixture = (event_id: string, hit: boolean) => ({
  event_id, team_home: "Tottenham", team_away: "Aston Villa", commence_time: "2026-09-19T10:30:00Z",
  predicted_scoreline: "1-1", actual_goals_home: 2, actual_goals_away: 3,
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  actual_outcome: "away_win" as const, hit, backfilled: true,
});

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
      { event_id: "e1", team_home: "A", team_away: "B", gameweek: 5, hit: true,
        made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e2", team_home: "C", team_away: "D", gameweek: 5, hit: true,
        made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e3", team_home: "E", team_away: "F", gameweek: 5, hit: false,
        made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
    ],
  },
  biggest_upsets: [],
  gameweeks: [{
    gameweek: 5, pct_correct: 2 / 3, n_fixtures: 3, n_rebuilt: 3,
    pct_correct_by_market: { exact_score: 0, match_result: 2 / 3, over_under_2_5: 1 / 3, btts: 1 },
    fixtures: [fixture("e1", true), fixture("e2", true), fixture("e3", false)],
  }],
};

// 3 counted picks, one of them made in time. The two figures genuinely differ,
// so a panel that rendered one number twice fails here.
const mixed: TrackRecordResponse = {
  ...allLate,
  summary: {
    ...allLate.summary,
    n_resolved_fixtures: 3, n_rebuilt_fixtures: 2, pct_correct_overall: 1 / 3,
    pct_correct_current_gameweek: 1 / 3, n_fixtures_current_gameweek: 3,
    pre_kickoff: {
      n_resolved_fixtures: 1, pct_correct_overall: 0, n_rebuilt_fixtures: 0,
      by_market: {
        exact_score: { pct_correct: 0, n_resolved: 1 },
        match_result: { pct_correct: 0, n_resolved: 1 },
        over_under_2_5: { pct_correct: 0, n_resolved: 1 },
        btts: { pct_correct: 0, n_resolved: 1 },
      },
    },
    per_pick: [
      { event_id: "e1", team_home: "A", team_away: "B", gameweek: 5, hit: false,
        made_before_kickoff: true, snapshotted_at: "2026-09-19T08:00:00Z", rebuilt: false },
      { event_id: "e2", team_home: "C", team_away: "D", gameweek: 5, hit: false,
        made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
      { event_id: "e3", team_home: "E", team_away: "F", gameweek: 5, hit: true,
        made_before_kickoff: false, snapshotted_at: "2026-09-19T12:00:00Z", rebuilt: true },
    ],
  },
};

// Exact strings, not regexes: the panel says "made before kickoff" in the card
// label AND in the sentence beneath it, so a regex matches twice and
// getByText throws on multiple matches. The exact label is unique.
const card = (label: string) => screen.getByText(label).parentElement;

describe("TrackRecordPanel counts every recorded pick", () => {
  it("puts the whole record in the headline, even when every pick was made late", () => {
    // This is the assertion the reversal exists for. The old panel rendered "—"
    // here and the sentence "Every pick below was made after kickoff and none
    // of them is counted", while 3 graded picks sat in the same payload.
    render(<TrackRecordPanel data={allLate} />);
    expect(card("Correct overall")).toHaveTextContent("67%");
    expect(card("Correct overall")).toHaveTextContent("2/3");
    expect(card("Correct overall")).not.toHaveTextContent("—");
  });

  it("shows the made-before-kickoff subset beside it, with its own n", () => {
    render(<TrackRecordPanel data={mixed} />);
    expect(card("Made before kickoff")).toHaveTextContent("0%");
    expect(card("Made before kickoff")).toHaveTextContent("0/1");
    // And it is a DIFFERENT number from the headline, which is the reason both
    // are published. If they ever render the same figure, the two-figure split
    // has been undone.
    expect(card("Correct overall")).toHaveTextContent("1/3");
  });

  it("reads as an absence, not 0%, when nothing was picked in time", () => {
    // 0 out of 0 is not 0%. It is the absence of a pre-game record, and it must
    // say so without withholding the headline, which is 2/3.
    render(<TrackRecordPanel data={allLate} />);
    expect(card("Made before kickoff")).toHaveTextContent("—");
    expect(card("Made before kickoff")).not.toHaveTextContent("0%");
  });

  it("never tells a reader a counted pick was held out of the record", () => {
    render(<TrackRecordPanel data={allLate} />);
    expect(document.body.textContent).not.toMatch(/not\s+counted/i);
    expect(document.body.textContent).not.toMatch(/not in the headline/i);
    expect(document.body.textContent).not.toMatch(/Rebuilt after/i);
  });

  it("discloses that its picks were made after kickoff", () => {
    // Disclosure, not exclusion. A reader looking at 2/3 needs to know all three
    // picks were made after the start; that is what the secondary figure is for,
    // and it is the sentence that says so — the per-fixture badges say it per card.
    render(<TrackRecordPanel data={allLate} />);
    expect(screen.getByText(/3 picks made after kickoff/i)).toBeInTheDocument();
    expect(screen.getByText(/None of them were made before kickoff/i)).toBeInTheDocument();
  });

  it("renders no NaN anywhere", () => {
    // Kept as its own test because it is the crash this state invites: a null
    // rate multiplied by a count.
    render(<TrackRecordPanel data={allLate} />);
    expect(document.body.textContent).not.toMatch(/NaN/);
  });

  it("still renders a panel when the payload predates the pre_kickoff field", () => {
    // A public_snapshot.json baked before the change can be live for minutes
    // after a deploy. A missing field must not crash the page — it crashed the
    // whole Data Hub the last time `by_market` was missing.
    const { pre_kickoff: _drop, ...summary } = allLate.summary;
    render(<TrackRecordPanel data={{ ...allLate, summary }} />);
    expect(card("Correct overall")).toHaveTextContent("2/3");
  });

  it("still lists the picks, so the headline is auditable", () => {
    render(<TrackRecordPanel data={allLate} />);
    expect(screen.getAllByText("Made after kickoff").length).toBeGreaterThan(0);
  });
});
