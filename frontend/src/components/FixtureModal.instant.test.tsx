import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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
  render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
  await screen.findByTestId("fixture-flow");
  return explain;
}

describe("FixtureModal's instant block, before any request", () => {
  it("renders the block from the bundle with the network blocked", async () => {
    mockApi();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network blocked"));
    const explain = await openModal();

    const block = screen.getByTestId("instant-block");
    expect(within(block).getByTestId("tile-result")).toBeInTheDocument();
    expect(within(block).getByTestId("pbar-fill")).toBeInTheDocument();
    expect(screen.getByText("Tottenham is the pick.")).toBeInTheDocument();
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
    expect(screen.queryByText("Rebuilt after kickoff")).toBeNull();
  });

  it("shows the rebuilt badge for a reconstructed review, and says the pick is not counted", async () => {
    mockApi();
    await openModal(finished("reconstructed"));

    expect(screen.getByText("Rebuilt after kickoff")).toBeInTheDocument();
    expect(screen.getByText(/not counted/i)).toBeInTheDocument();
    // The old line under the review claimed the opposite -- that a rebuilt pick
    // is "counted in the track record like any other pick". It is not, and two
    // claims about the same count may not differ.
    expect(screen.queryByText(/Counted in the track record/i)).toBeNull();
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

  it("names the pick from the stored review when the summary field is absent, and never scores a different side", async () => {
    mockApi();
    // 30 of the 380 fixture details in the shipped public snapshot predate the
    // `predicted_result` field the summary carries. The stored review names the
    // same three sides, so the pick is read from there rather than denied.
    const { predicted_result: _dropped, ...withoutPick } = finished("reconstructed");
    await openModal(withoutPick as unknown as FixtureDetail);

    expect(screen.getByText("Tottenham is the pick.")).toBeInTheDocument();
    expect(screen.queryByText(/No pick was made/i)).toBeNull();
    // The review's hit flag is about ITS pick, and here they are the same side.
    expect(screen.getByText(/the model's pick was right/i)).toBeInTheDocument();
    // The scoreline model can promote a draw the stored review did not pick.
    // When the two name different sides, no rightness is stated at all.
    vi.mocked(api.fixtureDetail).mockResolvedValue({
      ...withoutPick,
      predicted_result: "draw",
      post_match: { final_score: "1-1", provenance: "reconstructed", verdicts: REVIEW_VERDICTS, player_calls: [] },
    } as unknown as FixtureDetail);
    render(<FixtureModal eventId="e2" onClose={() => {}} explain={vi.fn().mockReturnValue(new Promise(() => {}))} />);
    await screen.findByTestId("instant-block");
    expect(screen.getByText("Draw is the pick.")).toBeInTheDocument();
    expect(screen.queryByText(/the model's pick was (right|wrong)/i)).toBeNull();
  });
});
