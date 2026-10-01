import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import type { FixtureDetail, PostMatchVerdict, TrackRecordResponse } from "../types";

// The parity rule for the instant block: everything it states must already be
// on this page before a reader spends a request. `api` is mocked here, so the
// only thing that could reach the network is `fetch` itself -- every test below
// blocks it and asserts it was never called.
vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
    trackRecord: vi.fn(),
  },
}));

const edge = (p: number, implied: number | null = null) => ({ prob: p, implied, edge: null });
const ou = (over: number) => ({ lambda_: 2.7, line: 2.5, over, under: 1 - over });

const REVIEW_VERDICTS: PostMatchVerdict[] = [
  { label: "Exact score", prediction: "2-1", actual: "1-0", hit: false },
  { label: "Match result", prediction: "home_win", actual: "home_win", hit: true },
  { label: "Goals O/U 2.5", prediction: "over", actual: "under", hit: false },
  { label: "BTTS", prediction: "yes", actual: "no", hit: false },
];

/** The site's own three-way figures. `implied` is present on all three, so the
 *  market row under the bar has an outcome above every one of its own. */
const detail = {
  event_id: "e1", commence_time: "2026-10-10T14:00:00Z",
  team_home: "Tottenham", team_away: "Aston Villa",
  home_win: edge(0.44, 0.47), draw: edge(0.28, 0.27), away_win: edge(0.28, 0.26),
  over_2_5: ou(0.52), under_2_5: ou(0.48), btts_yes_prob: 0.48,
  value_bet_flags: [], value_bet: null, has_live_odds: false,
  corners: ou(0.5), cards: ou(0.5),
  top_scoreline: "2-1", predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  home_context: { rest_days: 3, xg_for_last_5: 1.6, xg_against_last_5: 1.1, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.2 },
  away_context: { rest_days: 3, xg_for_last_5: 1.4, xg_against_last_5: 1.2, corners_last_5: 4, cards_last_5: 2, set_piece_xg_share_last_5: 0.18 },
  // ScorelineHeatmap indexes grid[0] unconditionally, so an empty grid is not a
  // valid FixtureDetail -- the real endpoint always returns rows.
  score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
  top_scorelines: [{ home: 2, away: 1, prob: 0.14 }],
  home_shots: 13.1, away_shots: 9.4, home_shots_on_target: 4.8, away_shots_on_target: 3.2,
  head_to_head: [], home_recent_form: [], away_recent_form: [],
  predicted_total_goals: 2.7, predicted_margin: 0.2,
  home_2plus_prob: 0.55, away_2plus_prob: 0.5,
  odds_fetched_at: null, odds_is_stale: false, recommended_bet: null,
  post_match: null, actual_stats: null, pre_match_value_bets: [],
} as unknown as FixtureDetail;

const finished = (provenance: "snapshot" | "reconstructed") => ({
  ...detail,
  commence_time: "2026-09-19T10:30:00Z",
  post_match: { final_score: "1-0", provenance, verdicts: REVIEW_VERDICTS, player_calls: [] },
});

/** `summary.n_resolved_fixtures` and `pct_correct_overall` are the two fields
 *  the strip reads. Both are scoped to picks made BEFORE kickoff -- the
 *  `TrackRecordSummary` comment at types.ts:517-521 says the headline counts
 *  only those, so a pick rebuilt after the match can never inflate it. */
const trackRecord = (n_resolved_fixtures: number, pct_correct_overall: number | null) =>
  ({ summary: { n_resolved_fixtures, pct_correct_overall } }) as unknown as TrackRecordResponse;

function mockApi(over: Record<string, unknown> = {}) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(detail);
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  vi.mocked(api.trackRecord).mockResolvedValue(trackRecord(71, 0.535));
  Object.assign(api, over);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

/** The block, once the modal's own fixture detail has landed. */
async function openModal(fixture: FixtureDetail = detail) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(fixture);
  const explain = vi.fn().mockReturnValue(new Promise(() => {}));
  const out = render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
  await screen.findByTestId("fixture-flow");
  return { explain, ...out };
}

describe("FixtureModal's instant block, before any request", () => {
  it("renders the block from the bundle with the network blocked", async () => {
    mockApi();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network blocked"));
    const { explain } = await openModal();

    const block = screen.getByTestId("instant-block");
    expect(within(block).getByTestId("tile-result")).toBeInTheDocument();
    // One bar, three segments: this site is a three-way market and the bar is
    // drawn three ways, so a single-segment assertion would pass on a bar that
    // dropped the draw.
    expect(within(block).getAllByTestId("pbar-fill")).toHaveLength(3);
    expect(within(block).getAllByTestId("pbar-label")).toHaveLength(3);
    expect(within(block).getByText("Tottenham is the pick.")).toBeInTheDocument();
    // The point of the block: none of the above cost anything.
    expect(explain).not.toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("keeps the three-way legend: the market row draws only when it covers every outcome", async () => {
    mockApi();
    await openModal();
    const block = screen.getByTestId("instant-block");
    // Three model segments, three market segments, and the row between them.
    expect(within(block).getAllByTestId("pbar-fill")).toHaveLength(3);
    expect(within(block).getByTestId("pbar-legend")).toBeInTheDocument();
    expect(within(block).getAllByTestId("pbar-market-fill")).toHaveLength(3);
    const figures = within(block).getByTestId("pbar-market-figures");
    expect(figures).toHaveTextContent("Tottenham 47%");
    // The draw keeps its own column; a three-way market drawn as two ways is a
    // lie by layout.
    expect(figures).toHaveTextContent("Draw 27%");
    expect(figures).toHaveTextContent("Aston Villa 26%");
  });

  it("shows a pre-match snapshot as the block's quiet chip, not an ad-hoc one", async () => {
    mockApi();
    await openModal(finished("snapshot"));

    // A stored pre-match snapshot IS a pick made before kickoff, so the block
    // says so once, in its own words.
    expect(screen.getByText("Made before kickoff")).toBeInTheDocument();
    // The two ad-hoc chips this replaces: the timing is now stated once, by the
    // block that governs how every figure below it should be read.
    expect(screen.queryByText("Pre-match snapshot")).toBeNull();
    expect(screen.queryByText("Made after kickoff")).toBeNull();
  });

  it("shows the timing badge for a reconstructed review, naming when the pick was made", async () => {
    mockApi();
    await openModal(finished("reconstructed"));

    // predictor-ui reworded this badge from "Rebuilt after kickoff" to
    // "Made after kickoff": since the track record began counting the earliest
    // recorded pick whatever moment it was made, when it was made is the honest
    // description of the state. The `rebuilt` key and its meaning are unchanged.
    expect(screen.getByText("Made after kickoff")).toBeInTheDocument();
  });

  it("no longer repeats the market bars below the panel, and keeps the figures that are not repeats", async () => {
    mockApi();
    await openModal();

    expect(screen.queryByText("Match result & goals")).toBeNull();
    // The bars the block already draws: 1x2, O/U 2.5 and BTTS are each on a
    // tile, a bar or a row that this modal still shows exactly once.
    expect(screen.queryByRole("img", { name: /BTTS: Yes/i })).toBeNull();
    // What the panel's extras do NOT carry stays: the corner, card, shots and
    // margin figures appear nowhere else on this page.
    expect(screen.getByText("Total corners")).toBeInTheDocument();
    expect(screen.getByText("Total cards")).toBeInTheDocument();
    expect(screen.getByText("Predicted margin")).toBeInTheDocument();
    expect(screen.getByText(/Predicted shots on target/)).toBeInTheDocument();
  });

  it("carries the record as hits over settled picks made before kickoff", async () => {
    mockApi();
    await openModal();

    // hits = Math.round(pct_correct_overall * n_resolved_fixtures) = 38,
    // settled = n_resolved_fixtures = 71. The strip prints the two counts and
    // never a derived percentage.
    expect(await screen.findByTestId("record-fill")).toBeInTheDocument();
    expect(screen.getByText("38/71")).toBeInTheDocument();
    expect(screen.getByText("Picks made before kickoff")).toBeInTheDocument();
    expect(screen.queryByText(/53\.5%/)).toBeNull();
  });

  it("shows no record strip when the track record has resolved nothing", async () => {
    mockApi();
    vi.mocked(api.trackRecord).mockResolvedValue(trackRecord(0, null));
    await openModal();

    const strip = (await screen.findByText("Picks made before kickoff")).parentElement!;
    await waitFor(() => expect(strip).toHaveTextContent("—"));
    // Nothing settled reads as a dash. "0/0" would be a claim about a record
    // that does not exist yet.
    expect(strip).not.toHaveTextContent("0/0");
    expect(screen.queryByTestId("record-fill")).toBeNull();
  });

  it("shows no record strip at all while the record is still loading, so it never flashes 0/0", async () => {
    mockApi();
    // Held pending: the record request is a real request this page makes, and
    // until it resolves there is no record to state. `record` is null in that
    // window by construction, so `InstantBlock` receives no `extras.record`.
    let resolveRecord: (v: TrackRecordResponse) => void = () => {};
    vi.mocked(api.trackRecord).mockReturnValue(
      new Promise<TrackRecordResponse>((r) => { resolveRecord = r; }) as never,
    );
    await openModal();

    // The block is on screen with no record row: absent, not "0/0" and not a dash.
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    expect(screen.queryByText("Picks made before kickoff")).toBeNull();
    expect(screen.queryByTestId("record-fill")).toBeNull();

    // And it arrives when it arrives, so the null above was a loading state and
    // not a fixture that carries no record.
    resolveRecord(trackRecord(71, 0.535));
    expect(await screen.findByTestId("record-fill")).toBeInTheDocument();
    expect(screen.getByText("38/71")).toBeInTheDocument();
  });

  it("accents the bar segment the bundle names, for all three outcomes", async () => {
    // The accent is resolved by EXACT label against the segments the shared
    // `panelFacts` adapter draws: home is `team_home`, away is `team_away` and the
    // draw is the bare word "Draw". `InstantBlock` passes the bundle's own pick
    // label straight through, so the site's job is to build `pick.label` in that
    // vocabulary -- a bundle whose `pick` was worded any other way would render a
    // bar with nothing accented at all, and no assertion here would notice.
    const ACCENT = "var(--color-pr-accent)";
    const cases: [string, string, number][] = [
      ["home_win", "Tottenham", 0],
      ["draw", "Draw", 1],
      ["away_win", "Aston Villa", 2],
    ];
    for (const [side, label, index] of cases) {
      // `mockApi()` per iteration: the previous modal's `afterEach` has not run
      // inside this loop, and reusing a resolved-mock after `restoreAllMocks`
      // between cases would leave `trackRecord` returning undefined.
      mockApi();
      const view = await openModal({ ...detail, predicted_result: side } as unknown as FixtureDetail);
      const block = screen.getByTestId("instant-block");
      const tones = within(block)
        .getAllByTestId("pbar-fill")
        .map((f) => (f as HTMLElement).style.backgroundColor);

      expect(tones, side).toHaveLength(3);
      // Exactly one accent, and it is the segment the bundle named.
      expect(tones.filter((t) => t === ACCENT), side).toHaveLength(1);
      expect(tones.indexOf(ACCENT), `${side} should accent segment ${index}`).toBe(index);
      // Said as well as painted, so a reader who cannot see the accent is told.
      expect(within(block).getByRole("img", { name: new RegExp(`the pick is ${label}`) })).toBeInTheDocument();
      expect(within(block).getByText(`${label} is the pick.`)).toBeInTheDocument();
      view.unmount();
    }
  });

  it("adds no second copy of any figure when the summary arrives", async () => {
    mockApi();
    // The block is the ONLY place the figures appear. Before the press it holds
    // one tile set, one bar, one legend and one record; the summary the button
    // fetches adds prose and nothing else, so every figure must still read
    // exactly once afterwards. Counting rather than presence-testing is the whole
    // assertion: a second copy is what presence-testing cannot see.
    const summary = {
      verdict: "Arsenal are the pick, and the market roughly agrees.",
      band: "moderate",
      pick: { label: "Tottenham win", side: "home_win" },
      factors: [
        { key: "result", direction: "neutral", headline: "Model and market agree", text: "They roughly do." },
      ],
      source: "template", model: "", generated_at: new Date().toISOString(), sport: "pl",
    };
    vi.mocked(api.fixtureDetail).mockResolvedValue(detail);
    const explain = vi.fn().mockResolvedValue(summary);
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    await screen.findByTestId("instant-block");

    const count = (testid: string) => screen.queryAllByTestId(testid).length;
    const before = {
      tiles: count("tile-result"),
      totalGoals: count("tile-total_goals"),
      btts: count("tile-btts"),
      fills: count("pbar-fill"),
      labels: count("pbar-label"),
      legends: count("pbar-legend"),
      marketFills: count("pbar-market-fill"),
      record: count("record-fill"),
    };
    // One of each while the summary is still behind the button. Every figure the
    // block states is enumerated here rather than sampled: a count over a
    // sampled list cannot see a figure that was duplicated and never sampled.
    expect(before).toEqual({
      tiles: 1, totalGoals: 1, btts: 1, fills: 3, labels: 3, legends: 1, marketFills: 3, record: 1,
    });

    fireEvent.click(screen.getByRole("button", { name: /ai summary/i }));
    await screen.findByTestId("fixture-summary");

    // Every figure still appears exactly once after the press. The block is the
    // only place they appear: the summary adds prose, not a second copy.
    expect({
      tiles: count("tile-result"),
      totalGoals: count("tile-total_goals"),
      btts: count("tile-btts"),
      fills: count("pbar-fill"),
      labels: count("pbar-label"),
      legends: count("pbar-legend"),
      marketFills: count("pbar-market-fill"),
      record: count("record-fill"),
    }).toEqual(before);
    // And the summary carries prose of its own, so "nothing was added" is not
    // trivially satisfied by an empty panel.
    expect(within(screen.getByTestId("fixture-summary")).getByText(summary.verdict)).toBeInTheDocument();
  });

  it("renders no shots or 2+ row for a fixture that carries neither, and never a NaN", async () => {
    mockApi();
    // PL#34, restated against the block. 30 of the 380 shipped details are
    // backfilled finished fixtures that carry NO `predicted_result`, NO
    // `home_shots`/`away_shots` and NO `home_2plus_prob`/`away_2plus_prob` key
    // at all. An ABSENT field is `undefined`, which is not `null`, so a `!== null`
    // guard waves it through to `undefined.toFixed(1)` -- which renders the
    // literal text "NaN" and, on a strict-mode React, blanks the whole modal.
    //
    // The guards are `hasNumber(...)`, so absent and null land in the same
    // honest branch: the row is left out. Nothing is fabricated, nothing is
    // coerced to 0, and no `NaN` reaches the page.
    const bare = {
      ...finished("reconstructed"),
      predicted_result: undefined,
      home_shots: undefined, away_shots: undefined,
      home_shots_on_target: undefined, away_shots_on_target: undefined,
      home_2plus_prob: undefined, away_2plus_prob: undefined,
    };
    const { predicted_result: _p, home_shots: _hs, away_shots: _as, home_shots_on_target: _hst,
      away_shots_on_target: _ast, home_2plus_prob: _h2, away_2plus_prob: _a2, ...absent } = bare;
    await openModal(absent as unknown as FixtureDetail);

    // It rendered: the pick came from the stored review, so the block still
    // states a fact rather than blanking.
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    expect(screen.getByText("Tottenham is the pick.")).toBeInTheDocument();
    // The rows with nothing to show are left out entirely.
    expect(screen.queryByText("Predicted shots")).toBeNull();
    expect(screen.queryByText("Predicted shots on target")).toBeNull();
    expect(screen.queryByText(/Tottenham to score 2\+/)).toBeNull();
    expect(screen.queryByText(/Aston Villa to score 2\+/)).toBeNull();
    // The figures that DO exist are unaffected -- an absent row must not take
    // its neighbours with it.
    expect(screen.getByText("Total corners")).toBeInTheDocument();
    expect(screen.getByText("Total cards")).toBeInTheDocument();
    // And not one `NaN` anywhere in the rendered modal.
    expect(document.body.textContent).not.toMatch(/NaN/);
  });

  it("names the pick from the stored review, which is the pick the record judges", async () => {
    mockApi();
    // 30 of the 380 fixture details in the shipped public snapshot carry no
    // `predicted_result` key at all — all 30 of them finished, all 30
    // reconstructed. The stored review names the same three sides, so the pick
    // is read from there rather than denied.
    const { predicted_result: _dropped, ...withoutPick } = finished("reconstructed");
    const first = await openModal(withoutPick as unknown as FixtureDetail);

    expect(within(screen.getByTestId("instant-block")).getByText("Tottenham is the pick.")).toBeInTheDocument();
    expect(screen.queryByText(/No pick was made/i)).toBeNull();
    expect(screen.getByText(/the model's pick was right/i)).toBeInTheDocument();

    // And where the two records disagree, the review's side is the pick. The
    // scoreline model promotes a draw the tracked pick was not
    // (`predicted_result` is the display variant; the record judges the plain
    // argmax), and the shipped snapshot disagrees on 17 of the 20 finished
    // fixtures it can render — so naming the promoted side here would put a
    // third pick on a page whose review row and list card both say Tottenham.
    first.unmount();
    vi.mocked(api.fixtureDetail).mockResolvedValue({
      ...withoutPick,
      predicted_result: "draw",
    } as unknown as FixtureDetail);
    render(<FixtureModal eventId="e2" onClose={() => {}} explain={vi.fn().mockReturnValue(new Promise(() => {}))} />);
    await screen.findByTestId("instant-block");
    expect(screen.getByText("Tottenham is the pick.")).toBeInTheDocument();
    expect(screen.queryByText("Draw is the pick.")).toBeNull();
  });
});
