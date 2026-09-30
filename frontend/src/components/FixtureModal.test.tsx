import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import type { FixtureDetail } from "../types";

vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
  },
}));

const edge = (p: number) => ({ prob: p, implied: null, edge: null });
// corners/cards carry the OverUnder shape; over_2_5/under_2_5 are MarketEdge
// (prob/implied/edge) like home_win/draw/away_win, which is what the modal reads.
const ou = (over: number) => ({ lambda_: 2.7, line: 2.5, over, under: 1 - over });

const detail = {
  event_id: "e1", commence_time: "2026-09-19T10:30:00Z",
  team_home: "Tottenham", team_away: "Aston Villa",
  home_win: edge(0.44), draw: edge(0.28), away_win: edge(0.28),
  over_2_5: edge(0.52), under_2_5: edge(0.48), btts_yes_prob: 0.48,
  value_bet_flags: [], value_bet: null, has_live_odds: false,
  corners: ou(0.5), cards: ou(0.5),
  top_scoreline: "1-1", predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  home_context: { rest_days: 3, xg_for_last_5: 1.6, xg_against_last_5: 1.1, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.2 },
  away_context: { rest_days: 3, xg_for_last_5: 1.4, xg_against_last_5: 1.2, corners_last_5: 4, cards_last_5: 2, set_piece_xg_share_last_5: 0.18 },
  // ScorelineHeatmap indexes grid[0] unconditionally, so an empty grid is not a
  // valid FixtureDetail -- the real endpoint always returns rows.
  score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
  top_scorelines: [{ home: 1, away: 1, prob: 0.14 }],
  home_shots: null, away_shots: null, home_shots_on_target: null, away_shots_on_target: null,
  head_to_head: [], home_recent_form: [], away_recent_form: [],
  predicted_total_goals: 2.7, predicted_margin: 0.2,
  home_2plus_prob: 0.55, away_2plus_prob: 0.5,
  odds_fetched_at: null, odds_is_stale: false, recommended_bet: null,
  post_match: null, actual_stats: null, pre_match_value_bets: [],
} as unknown as FixtureDetail;

function mockApi(over: Record<string, unknown> = {}) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(detail);
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  Object.assign(api, over);
}

const answer = {
  verdict: "Tottenham are the pick, and the market roughly agrees.",
  band: "moderate",
  factors: [
    {
      key: "result",
      direction: "neutral",
      headline: "Model and market agree",
      text: "Both put Tottenham at about the same price.",
    },
  ],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "pl",
  pick_timing: "pre_kickoff" as const,
};

describe("FixtureModal and the plain-English panel", () => {
  it("fetches nothing until asked, then shows the summary for this fixture", async () => {
    mockApi();
    const explain = vi.fn().mockResolvedValue(answer);
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    expect(explain).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
    expect(await screen.findByText("Tottenham are the pick, and the market roughly agrees.")).toBeInTheDocument();
    expect(explain).toHaveBeenCalledWith("pl", "e1");
  });

  it("shows the flow and the fixture while the summary is still being written", async () => {
    mockApi();
    const explain = vi.fn().mockReturnValue(new Promise<typeof answer>(() => {}));
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
    // No skeleton: the flow is the thing on screen while the request is in
    // flight. The team name appears in several places in this modal, so assert
    // the fixture heading reached the screen rather than a first match.
    expect(screen.getByTestId("fixture-flow")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Writing…" })).toBeDisabled();
    expect(await screen.findByRole("heading", { name: /Scoreline/i })).toBeInTheDocument();
  });

  it("offers a retry that asks the explainer again", async () => {
    mockApi();
    const explain = vi.fn().mockRejectedValue(new Error("down"));
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
    const retry = await screen.findByRole("button", { name: "Try again" });
    // The flow is still on screen beside the retry, not an error panel.
    expect(screen.getByTestId("fixture-flow")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    explain.mockResolvedValue(answer);
    fireEvent.click(retry);
    expect(await screen.findByText("Tottenham are the pick, and the market roughly agrees.")).toBeInTheDocument();
    expect(explain).toHaveBeenCalledTimes(2);
  });

  it("shows the rebuilt status the service reports, in this site's words", async () => {
    mockApi();
    const explain = vi.fn().mockResolvedValue({ ...answer, pick_timing: "rebuilt" });
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
    expect(await screen.findByText("Rebuilt after kickoff")).toBeInTheDocument();
    expect(screen.getByText(/not counted/)).toBeInTheDocument();
  });

  it("leaves the modal usable when there is no explainer", async () => {
    mockApi();
    render(<FixtureModal eventId="e1" onClose={() => {}} />);
    expect(screen.queryByText("Writing the summary…")).not.toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /Scoreline/i })).toBeInTheDocument();
  });
});

// 30 of the 380 rows in data/public_snapshot.json are backfilled finished
// fixtures that carry NO predicted_result at all, and no shot projections
// either (no home_shots / away_shots / home_shots_on_target /
// away_shots_on_target / home_2plus_prob / away_2plus_prob keys). An absent
// key is `undefined`, which is not `null`, so a `!== null` guard waved it
// through and `undefined.toFixed(1)` blanked the page. These tests hold both
// empty shapes -- ABSENT and explicit null -- to the same honest outcome:
// the row is left out, nothing is invented, and the modal still renders.
describe("FixtureModal when the fixture carries no prediction of its own", () => {
  const backfilled = {
    ...detail,
    post_match: {
      final_score: "3-0",
      provenance: "reconstructed",
      verdicts: [
        { label: "Exact score", prediction: "2-0", actual: "3-0", hit: false },
        { label: "Match result", prediction: "home_win", actual: "home_win", hit: true },
      ],
    },
    predicted_total_goals: null,
    predicted_margin: null,
  };

  // The 30 backfilled rows: keys simply not there. Spreading `undefined`
  // would not model that, so each variant deletes the key outright.
  const withoutKeys = (keys: string[], overrides: Record<string, unknown> = {}) => {
    const clone: Record<string, unknown> = { ...backfilled, ...overrides };
    for (const key of keys) delete clone[key];
    return clone as unknown as FixtureDetail;
  };

  const absentKeys = [
    "predicted_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_2plus_prob",
    "away_2plus_prob",
    "draw_signal",
    "pre_match_value_bets",
  ];

  async function renderDetail(fixtureDetail: FixtureDetail) {
    vi.mocked(api.fixtureDetail).mockResolvedValue(fixtureDetail);
    vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
    vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
    render(<FixtureModal eventId="backfill" onClose={() => {}} />);
    // A heading that only renders once the detail has landed and survived
    // rendering: if the modal threw, nothing after this can resolve.
    return screen.findByRole("heading", { name: /Scoreline/i });
  }

  it("renders the modal when predicted_result is absent, not just null", async () => {
    await expect(renderDetail(withoutKeys(absentKeys))).resolves.toBeInTheDocument();
    // The rest of the fixture is still there: it did not blank, and it did
    // not swallow the data it does have.
    expect(screen.getByText("Prediction review, final 3–0")).toBeInTheDocument();
  });

  it("renders the modal when predicted_result is explicitly null", async () => {
    const nulled = withoutKeys([], {
      predicted_result: null,
      home_shots: null,
      away_shots: null,
      home_shots_on_target: null,
      away_shots_on_target: null,
      home_2plus_prob: null,
      away_2plus_prob: null,
      pre_match_value_bets: [],
    });
    await expect(renderDetail(nulled)).resolves.toBeInTheDocument();
    expect(screen.getByText("Prediction review, final 3–0")).toBeInTheDocument();
  });

  // The two shapes are different states but they must be indistinguishable in
  // the UI: absent and null both mean "the model said nothing", and neither
  // may become a number.
  it("shows absent and null identically, inventing neither", async () => {
    const nulled = withoutKeys([], {
      predicted_result: null,
      home_shots: null,
      away_shots: null,
      home_shots_on_target: null,
      away_shots_on_target: null,
      home_2plus_prob: null,
      away_2plus_prob: null,
      pre_match_value_bets: [],
    });

    vi.mocked(api.fixtureDetail).mockResolvedValue(withoutKeys(absentKeys));
    const first = render(<FixtureModal eventId="absent" onClose={() => {}} />);
    await screen.findByRole("heading", { name: /Scoreline/i });
    const absentText = first.container.textContent;
    first.unmount();

    vi.mocked(api.fixtureDetail).mockResolvedValue(nulled);
    const second = render(<FixtureModal eventId="null" onClose={() => {}} />);
    await screen.findByRole("heading", { name: /Scoreline/i });
    const nullText = second.container.textContent;

    expect(absentText).toBe(nullText);
    // No fabricated value anywhere: not NaN, and not a 0 standing in for
    // "no prediction recorded".
    expect(nullText).not.toMatch(/NaN/);
    expect(nullText).not.toMatch(/\b0\.0\b/);
    // And the shot rows, which have no data here, are left out rather than
    // filled in.
    expect(nullText).not.toMatch(/Predicted shots/);
  });

  it("still omits the pre-match value bet section when the key is absent", async () => {
    await renderDetail(withoutKeys(absentKeys));
    // `undefined?.length > 0` is false, so this section must take the
    // "no bet was recorded" branch, not blow up on the missing array.
    expect(
      screen.getByText(/No value bet qualified before kickoff/),
    ).toBeInTheDocument();
  });
});
