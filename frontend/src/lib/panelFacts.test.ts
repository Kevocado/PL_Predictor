/** The v2 panel's figures, mapped from PL's own fixture data.
 *
 *  The three-way market is the case worth testing hard: a draw is a real outcome,
 *  and getting it wrong does not look wrong. It renders as a bar that still sums
 *  to 100% and still has labels — it is just quietly a lie about a third of the
 *  result space. So most of this file is about the draw and about §13b.
 */
import { describe, expect, it } from "vitest";

import { panelFacts } from "./panelFacts";
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

  it("omits the market row's legend unless implied covers ALL THREE outcomes", () => {
    // §13b. The committed snapshot has `implied: null` for all 380 fixtures, so
    // in practice the row is omitted everywhere — which means the branch that
    // DRAWS it is the one that needs a test, or it ships unexercised.
    expect(panelFacts(fixture()).legend).toEqual([]);

    const twoSides = fixture({ home_win: edge(0.48, 0.44), away_win: edge(0.26, 0.3) });
    expect(panelFacts(twoSides).legend).toHaveLength(2);

    const all = fixture({
      home_win: edge(0.48, 0.44), draw: edge(0.26, 0.25), away_win: edge(0.26, 0.31),
    });
    expect(panelFacts(all).legend.map((s) => s.label)).toEqual(["Arsenal", "Draw", "Chelsea"]);
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
