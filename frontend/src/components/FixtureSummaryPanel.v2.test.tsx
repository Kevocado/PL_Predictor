/** The v2 panel as this site actually ships it: a real answer from the real
 *  service, through the real component, over the real fixture.
 *
 *  Everything asserted here is asserted on the RENDERED result — the background
 *  colour the browser would paint a segment, and the accessible name a screen
 *  reader would read — and never on a prop having been passed. A prop assertion
 *  is satisfied by a site that hands the panel a pick it cannot use, and that is
 *  precisely the shape of the defect this file exists to catch.
 *
 *  **The defect, in the shape it actually takes on this site.** The bar joins the
 *  pick to a segment BY LABEL. PL's `/facts` names the pick `"<team> win"`
 *  (`match_pick` in the PL API's facts module) and this site labels its segments
 *  with bare team names, so a home or away pick matched nothing and the bar
 *  rendered with no accented segment at all. That is the panel's *correct*
 *  rendering of a bundle with no pick, printed directly beneath a verdict
 *  sentence naming one. It type-checked, it built, and every other test in this
 *  repo passed while it shipped.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import type { Explanation, Factor, PickRef } from "../predictor-ui";
import type { FixtureDetail, FixtureSummary } from "../types";

vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
    trackRecord: vi.fn(),
  },
}));

const ACCENT = "var(--color-pr-accent)";

/** The record the modal now carries, left unresolved on purpose: this file is
 *  about the accent, the de-emphasis and the market row, and a resolved record
 *  would add a bar of its own and two counts to the panel these assertions do
 *  not account for. The strip's own arithmetic is pinned in
 *  `FixtureModal.instant.test.tsx`. */
const noRecordYet = { summary: { n_resolved_fixtures: 0, pct_correct_overall: null } };

/** What `ProbabilityBar` painted each segment, in order — the rendered
 *  emphasis, read off the elements the browser would colour. */
const fills = (scope: ParentNode) =>
  [...scope.querySelectorAll<HTMLElement>("[data-testid='pbar-fill']")].map(
    (f) => f.style.backgroundColor,
  );

/** Which segment carries the accent, by index. -1 when none does. */
const accentedAt = (scope: ParentNode) => fills(scope).indexOf(ACCENT);

/** The DE-EMPHASIS, which is a different mechanism from the accent and fails
 *  independently of it. `dim` in `ProbabilityBar` writes an opacity onto both
 *  the segment fills and the figures, and it is driven by the selected factor
 *  rather than by the pick — so a bar can be correctly un-accented and wrongly
 *  dimmed, and only reading the opacity says which happened. */
const opacityOf =
  (selector: string) =>
  (scope: ParentNode): string[] =>
    [...scope.querySelectorAll<HTMLElement>(selector)].map((e) => e.style.opacity);

const fillOpacity = opacityOf("[data-testid='pbar-fill']");
const labelOpacity = opacityOf("[data-testid='pbar-label']");

const edge = (prob: number, implied: number | null = null) => ({ prob, implied, edge: null });

/** The §7b fixture. The model has Arsenal 48 / Draw 26 / Chelsea 26; the
 *  market's own figures, where present, are 44 / 25 / 31 — every one different
 *  from every model figure, so a legend built from the wrong array cannot be
 *  mistaken for the right one. */
const fixture = (over: Partial<FixtureSummary> = {}): FixtureSummary =>
  ({
    event_id: "gw1", commence_time: "2026-11-08T15:00:00Z",
    team_home: "Arsenal", team_away: "Chelsea",
    home_win: edge(0.48), draw: edge(0.26), away_win: edge(0.26),
    over_2_5: edge(0.56), under_2_5: edge(0.44),
    btts_yes_prob: 0.61, top_scoreline: "2-1",
    predicted_result: "home_win", draw_signal: false,
    is_fallback_prediction: false, data_confidence: "established",
    predicted_total_goals: 2.7, predicted_margin: 0.4,
    home_2plus_prob: 0.58, away_2plus_prob: 0.5, value_bet_flags: [],
    ...over,
  }) as FixtureSummary;

/** A detail the modal can open: the §7b fixture plus the fields the rest of
 *  the modal reads unconditionally (the heatmap indexes the grid, the context
 *  rows read both sides). Post-match stays null: the panel suite is about the
 *  pre-game panel, and a finished fixture belongs to the review tests. */
const modalDetail = (fx: FixtureSummary): FixtureDetail =>
  ({
    ...fx,
    over_2_5: { lambda_: 2.7, line: 2.5, over: 0.52, under: 0.48 },
    under_2_5: { lambda_: 2.3, line: 2.5, over: 0.48, under: 0.52 },
    value_bet_flags: [],
    value_bet: null,
    has_live_odds: false,
    corners: { lambda_: 10.2, line: 9.5, over: 0.5, under: 0.5 },
    cards: { lambda_: 3.8, line: 3.5, over: 0.5, under: 0.5 },
    home_context: { rest_days: 3, xg_for_last_5: 1.6, xg_against_last_5: 1.1, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.2 },
    away_context: { rest_days: 3, xg_for_last_5: 1.4, xg_against_last_5: 1.2, corners_last_5: 4, cards_last_5: 2, set_piece_xg_share_last_5: 0.18 },
    score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
    top_scorelines: [{ home: 1, away: 1, prob: 0.14 }],
    home_shots: null,
    away_shots: null,
    home_shots_on_target: null,
    away_shots_on_target: null,
    head_to_head: [],
    home_recent_form: [],
    away_recent_form: [],
    odds_fetched_at: null,
    odds_is_stale: false,
    recommended_bet: null,
    post_match: null,
    actual_stats: null,
    pre_match_value_bets: [],
  }) as unknown as FixtureDetail;

/** The block, as opposed to the whole modal — the ONLY place the figures appear.
 *
 *  **This helper used to scope to the plain-English panel, and that was the
 *  defect this phase removes.** The panel drew its own tiles, bar, legend and
 *  record from the same `extras` the block already rendered from, so every
 *  figure on this page was drawn twice; scoping by the panel's heading hid that
 *  rather than reporting it. `FixtureExplainer`'s summary state now passes the
 *  panel NO tiles, NO segments and NO record, so the block is the sole owner and
 *  every assertion in this file reads the block.
 *
 *  Scoping is still load-bearing, in the other direction: the modal renders other
 *  `role="img"` elements (team crests, the record strip) and other bars the
 *  site's own markup may add, so an unscoped query would quietly point at one of
 *  those and this whole file would pass while asserting about the wrong element. */
const panel = (container: HTMLElement) => {
  const found = container.querySelector<HTMLElement>("[data-testid='instant-block']");
  expect(found, "the instant block is not on the page").toBeTruthy();
  return found!;
};

/** The bar's graphic, scoped to that block, for the same reason. */
const panelImg = (container: HTMLElement) => {
  const bar = panel(container).querySelector<HTMLElement>("[role='img']");
  expect(bar, "the block drew no bar").toBeTruthy();
  return bar!;
};

/** A v2 answer in the shape the service sends. `pick` is spread in only when
 *  there is one, because the service states "no pick" by OMITTING the key and
 *  the panel acts on the absence. `factors` is overridable so a test can name a
 *  factor the panel draws no figure for. */
const v2 = (pick?: PickRef, factors?: Factor[]): Explanation =>
  ({
    verdict: "Arsenal are the pick, and the market roughly agrees.",
    band: "moderate",
    ...(pick ? { pick } : {}),
    factors: factors ?? [
      {
        key: "result",
        direction: "neutral",
        headline: "Model and market agree",
        text: "Both put Arsenal at about the same price.",
      },
    ],
    source: "template",
    model: "",
    generated_at: new Date().toISOString(),
    sport: "pl",
    pick_timing: "pre_kickoff",
  }) as Explanation;

/** Render the mounted panel and wait for the answer to land: the modal opens
 *  on the fixture detail (mocked at the api boundary, the way
 *  `FixtureModal.test.tsx` does), the flow renders from it with no request,
 *  and the summary the assertions below read sits behind the button. */
async function show(pick?: PickRef, fx: FixtureSummary = fixture(), factors?: Factor[]) {
  const explain = vi.fn().mockResolvedValue(v2(pick, factors));
  vi.mocked(api.fixtureDetail).mockResolvedValue(modalDetail(fx));
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  vi.mocked(api.trackRecord).mockResolvedValue(noRecordYet as never);
  const out = render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
  // Flow-first since the rollout: one press for the whole suite.
  fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
  await screen.findByText(/Arsenal are the pick/);
  return out;
}

describe("the accent follows the BUNDLE's pick, through this site's own labels", () => {
  /** The accent is resolved from the bundle the site hands the block, so the
   *  bundle is what these tests vary. The service's own `pick` is deliberately
   *  NOT the lever: the block renders before any request is made, so the summary
   *  cannot be what decides it.
   *
   *  The vocabulary still matters, and it is the site's to get right. The
   *  segments are labelled `Arsenal` / `Draw` / `Chelsea` by the shared
   *  `panelFacts` adapter, so a bundle whose `pick` was worded `"Arsenal win"`
   *  would join to nothing and render a bar with no accent at all — the panel's
   *  correct rendering of a bundle with no pick, printed under a verdict sentence
   *  naming one. `FixtureModal` builds `pick.label` in the segments' own
   *  vocabulary, and every case below is that join holding. */
  const cases: [string, number, string][] = [
    ["home_win", 0, "Arsenal"],
    ["draw", 1, "Draw"],
    ["away_win", 2, "Chelsea"],
  ];

  it("accents the pick's own segment, for all three outcomes of a three-way market", async () => {
    for (const [side, index, label] of cases) {
      const { container, unmount } = await show({ label: "Arsenal win" }, fixture({ predicted_result: side }));
      const tones = fills(panel(container));
      expect(tones, side).toHaveLength(3);
      // Exactly one segment is accented, and it is the one the bundle named.
      expect(tones.filter((t) => t === ACCENT), side).toHaveLength(1);
      expect(accentedAt(panel(container)), `${side} should accent segment ${index}`).toBe(index);
      expect(panelImg(container).getAttribute("aria-label"), side).toContain(`the pick is ${label}`);
      unmount();
    }
  });

  it("accents the second and third segments, not merely the first", async () => {
    // Stated on its own because this is the whole failure. A positional fill —
    // the mechanism the hub removed — passes every assertion above whenever the
    // pick happens to be the home side, because home is segment 0. These two are
    // the cases a positional implementation gets wrong, and a reader sees them as
    // the model having picked the side it picked least of.
    const away = await show(undefined, fixture({ predicted_result: "away_win" }));
    expect(accentedAt(panel(away.container))).toBe(2);
    away.unmount();

    const draw = await show(undefined, fixture({ predicted_result: "draw" }));
    expect(accentedAt(panel(draw.container))).toBe(1);
    draw.unmount();
  });

  it("is already accented before the button is pressed, so no request can change it", async () => {
    // The point of the block. The accent is a property of the bundle, so it is on
    // screen at zero requests and pressing the button does not move it — the
    // summary is prose, not a second opinion on which side was picked.
    const { container, unmount } = await show({ label: "Chelsea win", side: "away_win" }, fixture({ predicted_result: "home_win" }));
    expect(accentedAt(panel(container))).toBe(0);

    // And a service answer naming a DIFFERENT side cannot move it, because it is
    // not what the accent reads. If this ever starts following the summary, the
    // figures under a paid answer and the block above them disagree.
    expect(panelImg(container).getAttribute("aria-label")).toContain("the pick is Arsenal");
    unmount();
  });

  it("says which segment is the pick, for a reader who cannot see the accent", async () => {
    // The accent is a colour, so the emphasis also has to be said. This is the
    // assertion that catches a join failing for a reason the fill alone does not
    // explain: with no matched segment the accessible name simply has no "the
    // pick is" clause, and a screen-reader user is told nothing at all.
    const { container } = await show({ label: "Arsenal win" }, fixture({ predicted_result: "home_win" }));
    expect(panelImg(container)).toHaveAccessibleName(
      "Arsenal 48%, Draw 26%, Chelsea 26%, the pick is Arsenal",
    );
  });

  it("still accents nothing when the bundle genuinely carries no pick", async () => {
    // The control, and the reason the tests above mean anything: a bar that
    // emphasises something is claiming there is a pick. This is the exact shape
    // the defect impersonated, so it has to stay reachable and stay distinct.
    // `predicted_result` absent is not the same as null, and neither is a string
    // that names no side, so all three are exercised.
    for (const fx of [
      fixture({ predicted_result: undefined }),
      fixture({ predicted_result: null }),
      fixture({ predicted_result: "over_2_5" }),
    ] as unknown as FixtureSummary[]) {
      const { container, unmount } = await show({ label: "Arsenal win" }, fx);
      const tones = fills(panel(container));
      expect(tones).toHaveLength(3);
      expect(tones).not.toContain(ACCENT);
      // Three distinguishable tones, so the neutral ramp survives having no
      // accent spent on it — a reader must still be able to tell 48% from 26%.
      expect(new Set(tones).size).toBe(3);
      expect(panelImg(container)).toHaveAccessibleName("Arsenal 48%, Draw 26%, Chelsea 26%");
      // And the block says so in words rather than leaving the reader to infer it.
      expect(panel(container).textContent).toMatch(/No pick was made/i);
      unmount();
    }
  });
});

describe("the market row quotes the market, never the model", () => {
  it("draws the implied figures under the model's, and not the model's own", async () => {
    // §13b, and the one way this row can be silently defeated. A legend built
    // from the model's own probabilities renders a comparison of the model with
    // itself: two bars, the same numbers, the same widths, looking exactly like
    // a working market comparison. A presence test on the row passes while
    // shipping that, so the figures themselves are asserted — the market's
    // 44/25/31 against the model's 48/26/26, every figure different.
    const fx = fixture({
      home_win: edge(0.48, 0.44), draw: edge(0.26, 0.25), away_win: edge(0.26, 0.31),
    });
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fx);

    const row = panel(container).querySelector("[data-testid='pbar-market-figures']");
    expect(row).not.toBeNull();
    const quoted = [...row!.querySelectorAll<HTMLElement>("[data-market]")].map(
      (s) => s.textContent!.trim(),
    );
    expect(quoted).toEqual(["Arsenal 44%", "Draw 25%", "Chelsea 31%"]);

    // The model's figures appear once each, on the bar above, and NOT in the
    // market row. This is the assertion that fails if `legend` is ever handed
    // `segments` instead of `implied`.
    for (const model of ["48%", "26%"]) {
      expect(row!.textContent, `model figure ${model} leaked into the market row`).not.toContain(model);
    }
    // The row names itself, so it reads as the market's and not as more of the
    // model's — the labelling defect item 3 was about.
    expect(panel(container).querySelector("[data-testid='pbar-legend']")!.textContent).toContain("market");
  });

  it("omits the row when the market carries no implied figures, as today's snapshot does not", async () => {
    // Measured on the committed snapshot: `implied` is null for all 380 fixtures.
    // §13b says omit rather than draw a comparison the reader cannot make, and
    // omission has to be asserted at the rendered level too — a row of unlabelled
    // dashes is the defect the labelling fixed.
    const { container } = await show({ label: "Arsenal win", side: "home_win" });
    expect(panel(container).querySelector("[data-testid='pbar-legend']")).toBeNull();
    expect(panel(container).querySelector("[data-testid='pbar-market-figures']")).toBeNull();
  });

  it("omits the row when implied covers only two of the three outcomes", async () => {
    // A two-way legend under a three-way bar would sit under a draw segment with
    // nothing above it, inviting a comparison that cannot be made.
    const fx = fixture({ home_win: edge(0.48, 0.44), away_win: edge(0.26, 0.3) });
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fx);
    expect(panel(container).querySelector("[data-testid='pbar-legend']")).toBeNull();
  });
});

describe("an unplaceable pick fails closed, and the bar is never a control", () => {
  it("never hands the bar a pick it cannot place, so the accent always lands or is absent", async () => {
    // The fail-closed guarantee, stated as the site actually enforces it.
    //
    // `pickIndex` returns -1 for a label the bar cannot find, and the bar spends
    // its accent colour on no one rather than pointing at the nearest segment to
    // a claim nobody made. On this site that state is UNREACHABLE BY
    // CONSTRUCTION, and the construction is the thing worth pinning: `pick.label`
    // is built from the same `team_home` / `team_away` / `"Draw"` strings the
    // shared adapter labels its segments with, so a pick can only be named when
    // its side carries a probability — and a side with no probability has no
    // segment to name. Placement is a property of how the bundle is built, not
    // something resolved later and hoped for.
    //
    // The case that would break it is a bundle naming a side the adapter gave no
    // segment. So that is the case asserted: the draw has no probability, and
    // `predicted_result` still names the draw.
    const fx = fixture({
      draw: { prob: null, implied: null, edge: null },
      predicted_result: "draw",
    } as unknown as Partial<FixtureSummary>);
    const { container } = await show(undefined, fx);

    // Two segments, because there are two sides with a probability.
    const tones = fills(panel(container));
    expect(tones).toHaveLength(2);
    // And no pick, because the one the bundle named has nothing to point at.
    expect(tones).not.toContain(ACCENT);
    expect(panelImg(container).getAttribute("aria-label")).not.toMatch(/the pick is/);
    expect(panel(container).textContent).toMatch(/No pick was made/i);
  });

  it("leaves every figure at full opacity whatever the reader presses", async () => {
    // The block's bar is INERT: `InstantBlock` passes no `highlightKey` and no
    // `onSegmentFocus`, so the bar the reader sees before pressing anything
    // cannot be re-painted by a summary that has not been fetched yet. The
    // figures above the button are the truth, and nothing below it dims them.
    //
    // This used to be reachable: the panel drew its own bar, wired to the factor
    // list, and pressing a factor faded the others. The summary state no longer
    // renders a bar of its own, so there is nothing to fade — asserted as a
    // whole count, so a figure that stopped rendering an opacity at all fails
    // rather than passing as "not dimmed".
    const factors: Factor[] = [
      { key: "btts", direction: "neutral", headline: "Both score", text: "It often does." },
      { key: "record", direction: "neutral", headline: "Model record", text: "It has been good." },
    ];
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fixture(), factors);

    expect(fillOpacity(panel(container))).toEqual(["1", "1", "1"]);
    expect(labelOpacity(panel(container))).toEqual(["1", "1", "1"]);

    for (const key of ["btts", "record"]) {
      fireEvent.click(screen.getByTestId(`factor-${key}`));
      // Still nothing dimmed above the button, after either press.
      expect(fillOpacity(panel(container))).toEqual(["1", "1", "1"]);
      expect(labelOpacity(panel(container))).toEqual(["1", "1", "1"]);
    }

    // And the press is not a dead control either way: the summary panel is given
    // no tiles and no segments, so no factor key resolves to a figure and every
    // row reads as unpressed. That is the library's own rule — a key with no
    // figure under it clears the light rather than setting one — and asserting
    // it stops a future change from quietly re-wiring the summary's factors to
    // figures the reader is no longer shown twice.
    expect(screen.getByTestId("factor-btts")).toHaveAttribute("aria-pressed", "false");
    expect(panel(container).querySelector("[data-highlighted='true']")).toBeNull();
  });

  it("renders the segment figures as text, not as buttons with nothing behind them", async () => {
    // The bar sits above the button, in the state that is always true, and no
    // caller reports a segment focus — so `ProbabilityBar` renders each label as
    // text rather than as a focus stop that announces a control and does
    // nothing. A keyboard reader must not meet a control here.
    const { container } = await show({ label: "Arsenal win", side: "home_win" });
    for (const label of panel(container).querySelectorAll("[data-testid='pbar-label']")) {
      expect(label.tagName, "a segment label is a control with no listener").not.toBe("BUTTON");
    }
    // And the figures are still all there for a reader who takes no other route.
    expect(panel(container).textContent).toContain("Arsenal 48%");
    expect(panel(container).textContent).toContain("Chelsea 26%");
  });
});

describe("the flow stands alone when the explainer is unreachable", () => {
  it("shows the flow's text with no request made and no error in its place", async () => {
    const explain = vi.fn().mockRejectedValue(new Error("unreachable"));
    vi.mocked(api.fixtureDetail).mockResolvedValue(modalDetail(fixture()));
    vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
    vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
    vi.mocked(api.trackRecord).mockResolvedValue(noRecordYet as never);
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    // The flow renders from the detail with no request: the fixture's own
    // name, the three-way probabilities, the pick.
    expect(await screen.findByTestId("fixture-flow")).toBeInTheDocument();
    expect(screen.getByText("Arsenal vs Chelsea")).toBeInTheDocument();
    expect(explain).not.toHaveBeenCalled();
    // And the button offers the summary rather than an error taking its place.
    expect(screen.getByRole("button", { name: /ai summary/i })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByTestId("fixture-summary")).toBeNull();
  });
});
