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
  },
}));

const ACCENT = "var(--color-pr-accent)";

/** What `ProbabilityBar` painted each segment, in order — the rendered
 *  emphasis, read off the elements the browser would colour. */
const fills = (container: HTMLElement) =>
  [...container.querySelectorAll<HTMLElement>("[data-testid='pbar-fill']")].map(
    (f) => f.style.backgroundColor,
  );

/** Which segment carries the accent, by index. -1 when none does. */
const accentedAt = (container: HTMLElement) => fills(container).indexOf(ACCENT);

/** The DE-EMPHASIS, which is a different mechanism from the accent and fails
 *  independently of it. `dim` in `ProbabilityBar` writes an opacity onto both
 *  the segment fills and the figures, and it is driven by the selected factor
 *  rather than by the pick — so a bar can be correctly un-accented and wrongly
 *  dimmed, and only reading the opacity says which happened. */
const opacityOf =
  (selector: string) =>
  (container: HTMLElement): string[] =>
    [...container.querySelectorAll<HTMLElement>(selector)].map((e) => e.style.opacity);

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

/** The bar's graphic, scoped to the plain-English panel: the modal renders
 *  other `role="img"` elements (team badges), so a bare `getByRole("img")` is
 *  ambiguous at modal level and would quietly point at a logo. */
const panelImg = (container: HTMLElement) => {
  const panel = [...container.querySelectorAll("section")].find((s) =>
    s.textContent?.includes("In plain English"),
  );
  expect(panel, "the plain-English panel is not on the page").toBeTruthy();
  const bar = panel!.querySelector<HTMLElement>("[role='img']");
  expect(bar, "the panel drew no bar").toBeTruthy();
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
  const out = render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
  // Flow-first since the rollout: one press for the whole suite.
  fireEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
  await screen.findByText(/Arsenal are the pick/);
  return out;
}

describe("the accent follows the pick, through this site's own labels", () => {
  it("accents the pick's own segment, for all three outcomes of a three-way market", async () => {
    // The service's vocabulary, not a convenient one: `match_pick` words the two
    // sides "<team> win" and the draw "Draw". Home and away are the two that need
    // reconciling; the draw is the control, because it already matched before the
    // fix. That is exactly why the defect survived — two thirds of the market
    // rendered correctly, so a spot check found nothing.
    const cases: [PickRef, number, string][] = [
      [{ label: "Arsenal win", side: "home_win" }, 0, "Arsenal"],
      [{ label: "Draw", side: "draw" }, 1, "Draw"],
      [{ label: "Chelsea win", side: "away_win" }, 2, "Chelsea"],
    ];

    for (const [pick, index, label] of cases) {
      const { container, unmount } = await show(pick);
      const tones = fills(container);
      expect(tones, pick.label).toHaveLength(3);
      // Exactly one segment is accented, and it is the one the answer names.
      expect(tones.filter((t) => t === ACCENT), pick.label).toHaveLength(1);
      expect(accentedAt(container), `${pick.label} should accent segment ${index}`).toBe(index);
      expect(panelImg(container).getAttribute("aria-label"), pick.label).toContain(
        `the pick is ${label}`,
      );
      unmount();
    }
  });

  it("accents the second and third segments, not merely the first", async () => {
    // Stated on its own because this is the whole failure. A positional fill —
    // the mechanism the hub removed — passes every assertion above whenever the
    // pick happens to be the home side, because home is segment 0. These two are
    // the cases a positional implementation gets wrong, and a reader sees them as
    // the model having picked the side it picked least of.
    const away = await show({ label: "Chelsea win", side: "away_win" });
    expect(accentedAt(away.container)).toBe(2);
    away.unmount();

    const draw = await show({ label: "Draw", side: "draw" });
    expect(accentedAt(draw.container)).toBe(1);
    draw.unmount();
  });

  it("says which segment is the pick, for a reader who cannot see the accent", async () => {
    // The accent is a colour, so the emphasis also has to be said. This is the
    // assertion that catches a join failing for a reason the fill alone does not
    // explain: with no matched segment the accessible name simply has no "the
    // pick is" clause, and a screen-reader user is told nothing at all.
    const { container } = await show({ label: "Arsenal win", side: "home_win" });
    expect(panelImg(container)).toHaveAccessibleName(
      "Arsenal 48%, Draw 26%, Chelsea 26%, the pick is Arsenal",
    );
  });

  it("still accents nothing when the answer genuinely has no pick", async () => {
    // The control, and the reason the tests above mean anything: a bar that
    // emphasises something is claiming there is a pick. This is the exact shape
    // the defect impersonated, so it has to stay reachable and stay distinct.
    const { container } = await show(undefined);
    const tones = fills(container);
    expect(tones).toHaveLength(3);
    expect(tones).not.toContain(ACCENT);
    // Three distinguishable tones, so the neutral ramp survives having no accent
    // spent on it — a reader must still be able to tell a 48% from a 26%.
    expect(new Set(tones).size).toBe(3);
    expect(panelImg(container)).toHaveAccessibleName("Arsenal 48%, Draw 26%, Chelsea 26%");
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

    const row = container.querySelector("[data-testid='pbar-market-figures']");
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
    expect(container.querySelector("[data-testid='pbar-legend']")!.textContent).toContain("market");
  });

  it("omits the row when the market carries no implied figures, as today's snapshot does not", async () => {
    // Measured on the committed snapshot: `implied` is null for all 380 fixtures.
    // §13b says omit rather than draw a comparison the reader cannot make, and
    // omission has to be asserted at the rendered level too — a row of unlabelled
    // dashes is the defect the labelling fixed.
    const { container } = await show({ label: "Arsenal win", side: "home_win" });
    expect(container.querySelector("[data-testid='pbar-legend']")).toBeNull();
    expect(container.querySelector("[data-testid='pbar-market-figures']")).toBeNull();
  });

  it("omits the row when implied covers only two of the three outcomes", async () => {
    // A two-way legend under a three-way bar would sit under a draw segment with
    // nothing above it, inviting a comparison that cannot be made.
    const fx = fixture({ home_win: edge(0.48, 0.44), away_win: edge(0.26, 0.3) });
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fx);
    expect(container.querySelector("[data-testid='pbar-legend']")).toBeNull();
  });
});

describe("an unplaceable pick fails closed, in the accent AND in the de-emphasis", () => {
  it("accents no segment and dims no figure when the pick's label is no segment's label", async () => {
    // The fail-closed guarantee, stated in the bar's own terms and in BOTH of the
    // ways it can fail. `barPick` returns this label unchanged — branch 4 of its
    // own tests, in `panelFacts.test.ts` — so `pickIndex` finds nothing and the
    // bar spends its one accent colour on no one.
    //
    // The de-emphasis is a separate mechanism and has to be asserted separately.
    // It is driven by the SELECTED FACTOR, not by the pick, so a bar can be
    // correctly un-accented and wrongly dimmed at the same time, and no
    // assertion on `backgroundColor` can see that. "Arsenal to win" is the label
    // `barPick`'s own fail-closed test uses, so the two files agree on what an
    // unplaceable pick looks like rather than each inventing one.
    //
    // Why this needed saying: every other test in this file either selects no
    // factor — so nothing *can* be dimmed and a dim assertion would pass
    // vacuously — or asserts only the accent. The unplaceable-pick case had no
    // rendered-level test on this site at all, only the unit-level one, which is
    // exactly the gap a library change to `dim` or `pickIndex` would slide
    // through without turning anything red.
    const { container } = await show({ label: "Arsenal to win", side: "home_win" });

    // The accent half: no segment carries it, and nothing claims one in words.
    expect(fills(container)).toHaveLength(3);
    expect(accentedAt(container)).toBe(-1);
    expect(panelImg(container)).toHaveAccessibleName("Arsenal 48%, Draw 26%, Chelsea 26%");

    // The dim half: every figure is at full opacity, and the count is asserted
    // as a whole so a figure that stopped rendering an opacity at all fails
    // rather than passing as "not dimmed".
    expect(fillOpacity(container)).toEqual(["1", "1", "1"]);
    expect(labelOpacity(container)).toEqual(["1", "1", "1"]);
  });

  it("dims the other figures once a factor IS selected, so the assertion above is not vacuous", async () => {
    // The control, and the reason the test above means anything. A `dim` that
    // had stopped working entirely would ALSO render `["1","1","1"]` for an
    // unplaceable pick — the same class of silent breakage, in the opposite
    // direction, and equally invisible to a green suite. So the de-emphasis has
    // to be seen working somewhere, or its absence proves nothing.
    //
    // `btts` is the key that makes it work. Every segment carries the `result`
    // market, so a factor naming `result` dims nothing (the de-emphasis is per
    // MARKET, not per segment) — which is a fact worth pinning in its own right,
    // and is why this test uses a factor that names a TILE the bar does not draw.
    // `linkable` accepts it because a `btts` tile exists; `dim` then fades all
    // three segments, because none of them is about both-teams-to-score.
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fixture(), [
      { key: "btts", direction: "neutral", headline: "Both score", text: "It often does." },
    ]);
    fireEvent.click(screen.getByTestId("factor-btts"));

    // The de-emphasis is live, and it is the value `dim` actually writes.
    expect(fillOpacity(container)).toEqual(["0.4", "0.4", "0.4"]);
    expect(labelOpacity(container)).toEqual(["0.4", "0.4", "0.4"]);

    // The row that asked for the light is the row that is pressed, and the
    // highlight is announced as well as painted.
    expect(screen.getByTestId("factor-btts")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("factor-btts")).toHaveAttribute("data-highlighted", "true");
  });

  it("dims nothing at all when the selected factor names a figure the panel does not draw", async () => {
    // The other half of the linkage, and the one that was a defect until the
    // library made a key with no figure behind it CLEAR the highlight instead of
    // setting one. A factor is a reference to a figure; a key that resolves to
    // nothing used to be forwarded as though it did, and `dim` then faded every
    // figure in the panel with none lit — the worst of the three outcomes,
    // because the reader is left with less than before they pressed anything.
    //
    // This is not a rare shape: `template.py` always emits a `record` row and
    // pads with `context`, and neither is a market, so on a no-pick panel every
    // row is unlinkable. `linkable()` in `ExplainerPanel` now turns such a press
    // into a light that goes off, which is a change the reader can see and undo.
    const unlinkable: Factor[] = [
      { key: "record", direction: "neutral", headline: "Model record", text: "It has been good." },
      {
        key: "result",
        direction: "up",
        headline: "Model and market agree",
        text: "Both put Arsenal at about the same price.",
      },
    ];
    const { container } = await show({ label: "Arsenal win", side: "home_win" }, fixture(), unlinkable);

    // Pressing it sets nothing, so nothing is dimmed and nothing is lit.
    fireEvent.click(screen.getByTestId("factor-record"));
    expect(fillOpacity(container)).toEqual(["1", "1", "1"]);
    expect(screen.getByTestId("factor-record")).toHaveAttribute("aria-pressed", "false");
    expect(container.querySelector("[data-highlighted='true']")).toBeNull();

    // And the row that DOES name a drawn figure still lights it, so the test
    // above is about the key resolving rather than about selection being inert.
    fireEvent.click(screen.getByTestId("factor-result"));
    expect(screen.getByTestId("factor-result")).toHaveAttribute("aria-pressed", "true");
  });
});

describe("the flow stands alone when the explainer is unreachable", () => {
  it("shows the flow's text with no request made and no error in its place", async () => {
    const explain = vi.fn().mockRejectedValue(new Error("unreachable"));
    vi.mocked(api.fixtureDetail).mockResolvedValue(modalDetail(fixture()));
    vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
    vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
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
