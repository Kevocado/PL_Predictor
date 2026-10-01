/**
 * The reviewer's Phase-1 live follow-ups for PL, measured rather than eyeballed.
 *
 * 1. **A bare heading with nothing under it.** `FixtureFlow`'s pre-game row is
 *    the fixture's own name and nothing else, so before kickoff it rendered an
 *    `H4` reading `Tottenham vs Aston Villa` with no sentence beneath it,
 *    directly above the AI button. Measured per state, not assumed:
 *
 *      pre-game  — ONE row, and it is that heading. Nothing under it.
 *      in-play   — `FixtureDetail` carries no live score (its only score is
 *                  `post_match.final_score`), so a fixture that has kicked off
 *                  but has no `post_match` yet takes the SAME pre-game branch
 *                  and renders the SAME bare heading. Measured, not assumed.
 *      finished  — TWO real sentences (the result, and the pick's rightness).
 *                  These are live content and they must survive.
 *
 *    So this is NOT a blanket deletion of the flow. It is a heading that is
 *    conditional on rows existing beneath it, and the per-state behaviour is
 *    asserted here rather than checked by looking.
 *
 * 2. **One figure, one place.** The audit below is a field-by-field count of
 *    every figure-shaped token on the rendered modal, split by whether it sits
 *    inside the instant block or outside it. MEASURED RESULT: no figure is
 *    printed in both places, so nothing is deleted for 1b. PL's own Phase 1
 *    already removed the market bars (see the comment above the scoreline
 *    section in `FixtureModal.tsx`), which is why this repo measures clean
 *    where NBA's did not. The audit is pinned as assertions so a later change
 *    that reintroduces a second copy fails here instead of shipping.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import type { FixtureDetail, PostMatchVerdict, TrackRecordResponse } from "../types";

vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
    trackRecord: vi.fn(),
  },
}));

const edge = (p: number, implied: number | null = null) => ({ prob: p, implied, edge: null });
/** Every lambda/line/probability is a DISTINCT value on purpose.
 *
 *  A first pass of this audit reused one number for `predicted_total_goals` and
 *  the corners lambda, and set `cards.over` to the same decimal as
 *  `home_win.prob`. Both made the audit unable to tell the block's `total
 *  goals` tile from the site's `exp.` corners row, and reported a duplicate
 *  that did not exist while hiding the ones that did. A duplication audit is
 *  only worth anything when two different figures cannot collide by accident. */
const ou = (lambda_: number, line: number, over: number) => ({ lambda_, line, over, under: 1 - over });

const REVIEW_VERDICTS: PostMatchVerdict[] = [
  { label: "Exact score", prediction: "2-1", actual: "1-0", hit: false },
  { label: "Match result", prediction: "home_win", actual: "home_win", hit: true },
];

const preGame = {
  event_id: "e1", commence_time: "2026-10-10T14:00:00Z",
  team_home: "Tottenham", team_away: "Aston Villa",
  home_win: edge(0.44, 0.47), draw: edge(0.28, 0.27), away_win: edge(0.28, 0.26),
  over_2_5: ou(3.9, 3.5, 0.52), under_2_5: ou(3.9, 3.5, 0.48), btts_yes_prob: 0.48,
  value_bet_flags: [], value_bet: null, has_live_odds: false,
  corners: ou(5.4, 5.5, 0.61), cards: ou(2.1, 2.5, 0.72),
  top_scoreline: "2-1", predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  home_context: { rest_days: 3, xg_for_last_5: 1.63, xg_against_last_5: 1.12, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.21 },
  // `xg_for_last_5` is 1.73 and not 1.4: this file counts figures as SUBSTRINGS
  // of the rendered text, and "1.4" is a prefix of "1.47" as well as of "1.4",
  // so a value that merely looks different is not enough -- the digits have to
  // not appear inside any other rendered figure.
  away_context: { rest_days: 4, xg_for_last_5: 1.73, xg_against_last_5: 1.28, corners_last_5: 6, cards_last_5: 3, set_piece_xg_share_last_5: 0.17 },
  score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
  top_scorelines: [{ home: 2, away: 1, prob: 0.14 }],
  home_shots: 13.1, away_shots: 9.4, home_shots_on_target: 4.8, away_shots_on_target: 3.2,
  head_to_head: [], home_recent_form: [], away_recent_form: [],
  predicted_total_goals: 3.85, predicted_margin: 1.4,
  home_2plus_prob: 0.55, away_2plus_prob: 0.5,
  odds_fetched_at: null, odds_is_stale: false, recommended_bet: null,
  post_match: null, actual_stats: null, pre_match_value_bets: [],
} as unknown as FixtureDetail;

/** A fixture that has kicked off but has no `post_match` yet -- the in-play
 *  window. `FixtureDetail` has no live score field, so this is the whole of
 *  what this site's data can express about a match in progress. */
const inPlay = { ...preGame, commence_time: "2020-01-01T14:00:00Z" } as FixtureDetail;

const finished = {
  ...preGame,
  commence_time: "2026-09-19T10:30:00Z",
  post_match: { final_score: "1-0", provenance: "snapshot", verdicts: REVIEW_VERDICTS, player_calls: [] },
} as unknown as FixtureDetail;

const trackRecord = { summary: { n_resolved_fixtures: 71, pct_correct_overall: 0.535 } } as unknown as TrackRecordResponse;

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

async function openModal(fixture: FixtureDetail, explain = vi.fn().mockReturnValue(new Promise(() => {}))) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(fixture);
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  vi.mocked(api.trackRecord).mockResolvedValue(trackRecord);
  const out = render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
  await screen.findByTestId("fixture-flow");
  return { explain, ...out };
}

/** The flow's rows, by tag and text -- so a bare heading is visible AS a
 *  heading rather than as a string that happens to be a team's name. */
function flowRows() {
  return Array.from(screen.getByTestId("fixture-flow").children).map((el) => `${el.tagName}: ${el.textContent}`);
}

/** How many times a figure appears somewhere. Counted over the whole document
 *  and, separately, over everything OUTSIDE the block: a duplicate is a
 *  duplicate wherever the second copy sits, and scoping the count to a
 *  component is how the original triplication passed. */
function outsideBlock() {
  const clone = document.body.cloneNode(true) as HTMLElement;
  // A real DOM region, not a string subtraction: subtracting the block's
  // textContent can also delete a coincidentally identical run elsewhere.
  for (const el of Array.from(clone.querySelectorAll('[data-testid="instant-block"]'))) el.remove();
  return clone.textContent ?? "";
}
const count = (needle: string, root: string) => root.split(needle).length - 1;

describe("the flow's name heading is conditional on rows existing beneath it", () => {
  it("renders no heading, and no bare flow, before kickoff", async () => {
    await openModal(preGame);
    const flow = screen.getByTestId("fixture-flow");

    // MEASURED BEFORE THE FIX: this flow had exactly one row, an
    // `H4: "Tottenham vs Aston Villa"`, with nothing under it.
    expect(flowRows()).toEqual([]);

    // Asserted as the absence of a HEADING, not as the absence of one team's
    // name, so renaming the fixture cannot let the bare heading back through.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(within(flow).queryByRole("heading")).toBeNull();
    expect(flow.textContent?.trim()).toBe("");

    // The facts are untouched: the block still names the pick. The heading
    // went; the block did not. `team_home` is carried for exactly this reason
    // -- `bundleFacts.fullTeamName` reads `home_team ?? team_home`.
    const block = screen.getByTestId("instant-block");
    expect(block).toBeInTheDocument();
    expect(within(block).getByText("Tottenham is the pick.")).toBeInTheDocument();
  });

  it("renders no bare heading in the in-play window either", async () => {
    // MEASURED BEFORE THE FIX: this state rendered the SAME bare
    // `H4: "Tottenham vs Aston Villa"`. `FixtureDetail` carries no live score,
    // so there is no score sentence for it to say, and the honest reading is
    // an empty flow -- the facts are above it in the block.
    await openModal(inPlay);
    const flow = screen.getByTestId("fixture-flow");

    expect(flowRows()).toEqual([]);
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
  });

  it("keeps the flow's real sentences once the fixture is finished", async () => {
    await openModal(finished);
    const flow = screen.getByTestId("fixture-flow");

    // MEASURED BEFORE THE FIX, and the reason this is not a blanket deletion:
    // the finished flow carries two genuine sentences. A fix that removed the
    // whole flow would have taken these with it.
    expect(flowRows()).toEqual([
      "P: The result is a win for Tottenham.",
      "P: The pick rightness: the model's pick was right.",
    ]);
    expect(flow).toHaveTextContent("The result is a win for Tottenham.");
    expect(flow).toHaveTextContent("The pick rightness: the model's pick was right.");

    // The finished rows are sentences, not a name, so there is no heading here
    // either -- and the result sentence still names the side, which is the
    // proof that withholding `home_team` pre-game did not take the names away
    // from the state that needs them.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
  });

  it("keeps the block's own figures in all three states", async () => {
    for (const [name, fixture] of [["pre-game", preGame], ["in-play", inPlay], ["finished", finished]] as const) {
      const { unmount } = await openModal(fixture);
      const block = screen.getByTestId("instant-block");
      expect(within(block).getByTestId("tile-result"), name).toHaveTextContent("44%");
      expect(within(block).getByTestId("tile-total_goals"), name).toHaveTextContent("3.9");
      expect(within(block).getByTestId("tile-btts"), name).toHaveTextContent("48%");
      // One bar, three segments: a three-way market drawn as two ways is a lie
      // by layout, so the draw keeps its own column.
      expect(within(block).getAllByTestId("pbar-fill"), name).toHaveLength(3);
      expect(within(block).getByText("Tottenham is the pick."), name).toBeInTheDocument();
      unmount();
    }
  });
});

describe("one figure, one place", () => {
  it("draws every block figure in the block and nowhere else", async () => {
    await openModal(preGame);
    const block = screen.getByTestId("instant-block");
    const blockText = block.textContent ?? "";
    const out = outsideBlock();

    // Each of these is a figure the instant block owns, so each must appear
    // ZERO times outside it.
    for (const figure of ["44%", "28%", "47%", "27%", "26%", "48%", "3.9"]) {
      expect(count(figure, blockText), `${figure} in block`).toBeGreaterThan(0);
      expect(count(figure, out), `${figure} outside block`).toBe(0);
    }

    // The market legend, the tile value and the bar label are all the BLOCK's
    // own three renderings of a figure it owns -- counted in the block, not
    // outside, and that is where the one-source rule puts them.
    expect(count("44%", blockText)).toBe(2); // the result tile + the bar's label
  });

  it("keeps each site figure the block does not draw, exactly once", async () => {
    await openModal(preGame);
    const blockText = screen.getByTestId("instant-block").textContent ?? "";
    const out = outsideBlock();

    // The block has NO tile for any of these, so the site's own sections are
    // their only source on the page. Deleting them wholesale would have removed
    // figures that exist nowhere else -- the opposite of a duplication rule,
    // which is about a figure appearing twice, not once. Each must be present
    // once, in the site's own section, and absent from the block.
    const siteOnly: Array<[string, string]> = [
      ["55%", "Tottenham to score 2+"],
      ["50%", "Aston Villa to score 2+"],
      ["61%", "Total corners over"],
      ["72%", "Total cards over"],
      ["1.4", "Predicted margin"],
      ["16.0%", "Scoreline heatmap's most probable cell"],
    ];
    for (const [figure, what] of siteOnly) {
      expect(count(figure, blockText), `${what} (${figure}) in block`).toBe(0);
      expect(count(figure, out), `${what} (${figure}) outside block`).toBe(1);
    }
  });

  it("finds no figure printed in both the block and the site at all", async () => {
    // The systematic form of the audit, and the reason this repo's PR changes
    // no figure: sweep EVERY figure on the rendered modal and fail on any that
    // sits in both regions. Figures are compared as WHOLE TOKENS, never as
    // substrings -- a substring count invents duplicates, because "7%" occurs
    // inside "17%" and "4%" inside "54%". On a page carrying a 17% draw, a 17%
    // assist and a 54% BTTS, the substring form of this sweep reports three
    // phantom duplications on a page that has none. (That is not hypothetical:
    // the first pass of this audit did exactly that.)
    //
    // A limit worth stating, because it was measured rather than assumed: two
    // DIFFERENT facts that round to the same number will still trip a numeric
    // sweep, and that is not a duplication. In the real Arsenal 0.2 Leeds
    // snapshot the bar's draw segment (0.168 -> 17%) and Stach's anytime-assist
    // probability (0.165 -> 17%) both read "17%", in two different sections,
    // about two different things. There is no second copy of the draw figure
    // anywhere on that page, which is what the field-identity assertions above
    // are for. So the sweep below is a guard against regression, not a claim
    // that equal numbers imply equal facts.
    for (const [name, fixture] of [["pre-game", preGame], ["finished", finished]] as const) {
      const { unmount } = await openModal(fixture);
      const blockText = screen.getByTestId("instant-block").textContent ?? "";
      const out = outsideBlock();
      const tokenise = (s: string) => s.match(/\d+(?:\.\d+)?%?/g) ?? [];
      const whole = document.body.textContent ?? "";
      const figures = [...new Set(tokenise(whole))].filter((t) => t.includes("%") || t.includes("."));
      const both = figures.filter(
        (t) => tokenise(blockText).filter((x) => x === t).length > 0 && tokenise(out).filter((x) => x === t).length > 0,
      );
      expect(both, `figures in both the block and the site (${name})`).toEqual([]);
      unmount();
    }
  });

  it("does not repeat the block's figures after the summary is pressed", async () => {
    const explain = vi.fn().mockResolvedValue({
      verdict: "Tottenham is the pick at home.",
      band: "moderate",
      factors: [{ key: "result", direction: "up", headline: "Home edge", text: "Tottenham at home." }],
      source: "template" as const,
      model: "",
      generated_at: new Date().toISOString(),
      sport: "pl",
      pick_timing: "pre_kickoff" as const,
    });
    await openModal(preGame, explain);
    // Press the real button, so the summary state is the one under test.
    const button = screen.getByRole("button", { name: /ai summary/i });
    (button as HTMLButtonElement).click();
    const summary = await screen.findByTestId("fixture-summary");

    // The block and the flow never unmount, and the summary draws no tile, bar
    // or record of its own -- so the whole-page count is unchanged in this state.
    expect(within(summary).queryByTestId("tile-result")).toBeNull();
    expect(within(summary).queryByTestId("tile-total_goals")).toBeNull();
    expect(count("44%", document.body.textContent ?? "")).toBe(2);
    expect(count("44%", outsideBlock())).toBe(0);
    expect(explain).toHaveBeenCalledTimes(1);
  });
});
