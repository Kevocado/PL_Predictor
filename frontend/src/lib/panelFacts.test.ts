/** The v2 panel's figures, mapped from PL's own fixture data.
 *
 *  The three-way market is the case worth testing hard: a draw is a real outcome,
 *  and getting it wrong does not look wrong. It renders as a bar that still sums
 *  to 100% and still has labels — it is just quietly a lie about a third of the
 *  result space. So most of this file is about the draw and about §13b.
 */
import { describe, expect, it } from "vitest";

import { barPick, panelFacts } from "./panelFacts";
import type { PickRef, Segment } from "../predictor-ui";
import type { FixtureSummary } from "../types";

const edge = (prob: number, implied: number | null = null) => ({ prob, implied, edge: null });

const fixture = (over: Partial<FixtureSummary> = {}): FixtureSummary => ({
  event_id: "gw1", commence_time: "2026-11-08T15:00:00Z",
  team_home: "Arsenal", team_away: "Chelsea",
  home_win: edge(0.48), draw: edge(0.26), away_win: edge(0.26),
  over_2_5: edge(0.56), under_2_5: edge(0.44),
  btts_yes_prob: 0.61,
  top_scoreline: "2-1",
  predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  predicted_total_goals: 2.7, predicted_margin: 0.4,
  home_2plus_prob: 0.58, away_2plus_prob: 0.5,
  value_bet_flags: [],
  ...over,
} as FixtureSummary);

describe("panelFacts", () => {
  it("maps the §7b PL fixture exactly", () => {
    const { tiles, segments } = panelFacts(fixture());
    expect(segments.map((s) => `${s.label} ${s.prob}`)).toEqual([
      "Arsenal 0.48", "Draw 0.26", "Chelsea 0.26",
    ]);
    expect(tiles.map((t) => `${t.market}:${t.value}`)).toEqual([
      "result:48%", "total_goals:2.7", "btts:61%",
    ]);
    expect(tiles[0].sub).toBe("win · Arsenal");
  });

  it("gives the draw its OWN segment rather than folding it into 'not Chelsea'", () => {
    // A three-way market rendered as two ways still sums to 100% and still has
    // labels, so it looks right and is wrong about a third of the result space.
    const { segments } = panelFacts(fixture());
    expect(segments).toHaveLength(3);
    expect(segments.map((s) => s.label)).toEqual(["Arsenal", "Draw", "Chelsea"]);
    expect(segments.reduce((sum, s) => sum + s.prob, 0)).toBeCloseTo(1, 6);
  });

  it("tags every segment with the result market, so a factor can highlight it", () => {
    expect(panelFacts(fixture()).segments.every((s) => s.market === "result")).toBe(true);
  });

  it("returns a legend only from the market's `implied`, and as far as `implied` carries", () => {
    // §13b. The committed snapshot has `implied: null` for all 380 fixtures, so
    // in practice the row is omitted everywhere — which means the branch that
    // DRAWS it is the one that needs a test, or it ships unexercised.
    //
    // What this function under test actually does is carry `implied` as far as
    // the facts carry it and no further. It omits NOTHING: a two-way fixture
    // returns a two-element legend, and the ROW is what disappears. The gate for
    // that is `covers()` in `ProbabilityBar`, which requires the legend to cover
    // the segments in both directions — see the row-level test in
    // `FixtureSummaryPanel.v2.test.tsx` ("omits the row when implied covers only
    // two of the three outcomes"). Asserting the omission here instead would
    // have been a test whose name claims this function drops a legend it keeps.
    expect(panelFacts(fixture()).legend).toEqual([]);

    // Two outcomes of `implied`, two legend entries: the mapping does not stop
    // at three-way to make a decision about the row.
    const twoSides = fixture({ home_win: edge(0.48, 0.44), away_win: edge(0.26, 0.3) });
    expect(panelFacts(twoSides).legend).toHaveLength(2);

    // And it never invents a figure to complete the set — see the test below.
    const all = fixture({
      home_win: edge(0.48, 0.44), draw: edge(0.26, 0.25), away_win: edge(0.26, 0.31),
    });
    expect(panelFacts(all).legend.map((s) => s.label)).toEqual(["Arsenal", "Draw", "Chelsea"]);
    // The market's own numbers, not the model's — the two are equal nowhere here.
    expect(panelFacts(all).legend.map((s) => s.prob)).toEqual([0.44, 0.25, 0.31]);
  });

  it("never derives a draw probability to fill a missing implied", () => {
    // 0.44 + 0.3 leaves 0.26 unaccounted for and subtraction "works". It is
    // still wrong: it assumes the overround is spread across all three outcomes,
    // and books spread it differently. A number this product computed rather
    // than one the facts carry is the line §1 draws.
    const partial = panelFacts(fixture({ home_win: edge(0.48, 0.44), away_win: edge(0.26, 0.3) }));
    expect(partial.legend.some((s) => s.label === "Draw")).toBe(false);
  });

  it("gives the result tile the LEADING outcome, even when that is the draw", () => {
    const drawLeads = fixture({ home_win: edge(0.2), draw: edge(0.45), away_win: edge(0.35) });
    const { tiles } = panelFacts(drawLeads);
    expect(tiles[0].value).toBe("45%");
    expect(tiles[0].sub).toBe("win · Draw");
  });

  it("treats a missing or nonsensical probability as absent, never as 0", () => {
    for (const bad of [undefined, null, 1.4, -0.2, Number.NaN]) {
      const out = panelFacts(fixture({
        home_win: edge(bad as never), draw: edge(0.26), away_win: edge(0.26),
      }));
      expect(out.segments.map((s) => s.label)).toEqual(["Draw", "Chelsea"]);
    }
  });

  it("omits a market the fixture does not carry", () => {
    const { tiles } = panelFacts(fixture({ predicted_total_goals: null, btts_yes_prob: 1.4 }));
    expect(tiles.map((t) => t.market)).toEqual(["result"]);
  });

  it("returns nothing at all before the fixture has loaded", () => {
    expect(panelFacts(null)).toEqual({ tiles: [], segments: [], legend: [] });
    expect(panelFacts(undefined)).toEqual({ tiles: [], segments: [], legend: [] });
  });
});

/** This site's own bar vocabulary: bare team names, and the draw as its own
 *  segment. `panelFacts` builds exactly this, and PL's `/facts` words the same
 *  sides `"<team> win"` — the join that disagrees is the whole reason `barPick`
 *  exists. */
const barSegments = (): Segment[] => [
  { label: "Arsenal", prob: 0.48, market: "result" },
  { label: "Draw", prob: 0.26, market: "result" },
  { label: "Chelsea", prob: 0.26, market: "result" },
];

describe("barPick", () => {
  it("returns a pick the bar already names, unchanged", () => {
    // Branch 1. The draw needs no translation — which is exactly why the defect
    // survived: two thirds of PL's market rendered correctly, so a spot check on
    // a draw fixture found nothing. This branch is the control the others are
    // measured against, and it must not be "improved" into stripping a suffix
    // the bar already agrees with.
    const pick: PickRef = { label: "Draw", side: "draw" };
    const out = barPick(pick, barSegments());
    expect(out.label).toBe("Draw");
    expect(out.side).toBe("draw");
  });

  it("returns a pick whose label this site does not suffix, unchanged", () => {
    // Branch 2. A label that is not a "win" pick has no suffix to strip, so
    // nothing may be stripped off it. This is the branch that keeps the suffix
    // rule from becoming a blind `slice`: the two cases below both end in four
    // characters, and if the suffix check were dropped, the first would be cut
    // down to exactly `"Arsenal"` and snapped onto the Arsenal segment.
    const segments = barSegments();

    // A scoreline pick, which PL's facts do carry (`top_scoreline`). The claim
    // is "Arsenal 2-1", which is NOT the claim "Arsenal" — and the pick's side is
    // not something this function gets to re-read off the label.
    const scoreline: PickRef = { label: "Arsenal 2-1", side: "home_win" };
    const cut = barPick(scoreline, segments);
    expect(cut.label).toBe("Arsenal 2-1");
    expect(cut.side).toBe("home_win");
    // Nothing accented, which is the point: slicing the suffix here would have
    // put the accent on a segment nobody picked.
    expect(segments.findIndex((s) => s.label === cut.label)).toBe(-1);

    // And a draw pick on a fixture whose draw probability is missing or
    // nonsense, so `panelFacts` never built a `"Draw"` segment. The pick is real
    // and the bar cannot show it, which is not this function's to fix by
    // guessing a side.
    const noDraw = segments.filter((s) => s.label !== "Draw");
    const draw: PickRef = { label: "Draw", side: "draw" };
    const out = barPick(draw, noDraw);
    expect(out.label).toBe("Draw");
    expect(out.side).toBe("draw");
    expect(noDraw.findIndex((s) => s.label === out.label)).toBe(-1);
  });

  it("translates a side pick into the label this site's bar is drawn in", () => {
    // Branch 3, and the fix. PL's `match_pick` says `"Arsenal win"`; this site's
    // segments say `"Arsenal"`. The label moves and nothing else does — the
    // pick's identity is the service's, and only the join key is the site's.
    for (const [label, side, want] of [
      ["Arsenal win", "home_win", "Arsenal"],
      ["Chelsea win", "away_win", "Chelsea"],
    ] as [string, string, string][]) {
      const out = barPick({ label, side }, barSegments());
      expect(out.label, label).toBe(want);
      // The side is carried through untouched: never re-derived from this
      // site's own probabilities, which is the defect the field exists to remove.
      expect(out.side, label).toBe(side);
    }
  });

  it("returns a pick it cannot place UNCHANGED, so the bar accents nothing", () => {
    // Branch 4, and the one that matters. The fail-closed guarantee: a label with
    // no segment behind it comes back byte-identical, so `pickIndex` finds
    // nothing and the bar spends its one accent colour on no one. Falling back to
    // the nearest — or the first — segment would put the emphasis on a claim
    // nobody made, which is the same silent wrongness as the defect `barPick`
    // exists to fix, reached by a different route. Nothing else in the suite
    // catches it: the component tests in `FixtureSummaryPanel.v2.test.tsx` only
    // use picks this bar CAN place, so they stay green under this mutation.
    const segments = barSegments();
    const before = structuredClone(segments);
    const pick: PickRef = { label: "Arsenal to win", side: "home_win" };
    const out = barPick(pick, segments);

    // Byte-identical to the input, not merely "close": no case folding, no
    // trimming, no substitution of the nearest team's name.
    expect(out.label).toBe("Arsenal to win");
    expect(out.label).toBe(pick.label);
    expect(out.side).toBe("home_win");
    // The consequence, stated in the bar's own terms: no segment matches, so
    // `pickIndex` is -1 and the accent is spent on nothing. This is the
    // fail-closed behaviour, and it is what a first-segment fallback would take.
    expect(segments.findIndex((s) => s.label === out.label)).toBe(-1);
    // A translation helper that edited its input would be corrupting the bar's
    // own labels, and the next render would be wrong in a new way.
    expect(segments).toEqual(before);
  });
});
