/** The figures the v2 panel draws, mapped from PL's own fixture data.
 *
 *  The three-way market is the whole difficulty here, and it is why this file
 *  exists rather than a one-liner in the panel.
 *
 *  **Measured on the committed snapshot, all 380 fixtures:** `implied` is
 *  `null` for every outcome of every fixture, and `has_live_odds` is `false`
 *  throughout. So the market row — the model's split against the book's — has
 *  nothing to draw against today, and §13b says omit it rather than draw a
 *  comparison the reader cannot make.
 *
 *  That is a fact about today's data, not a permanent one: the field is typed
 *  per-outcome (`MarketEdge.implied`), so a fixture with live odds would carry
 *  all three. The mapping therefore builds the legend when all three are present
 *  and lets `ProbabilityBar` omit the row otherwise, and the test below pins
 *  both branches so neither is untested — a rule that never fires is a rule
 *  nobody has checked.
 */
import { pct, stat } from "../predictor-ui";
import type { MarketTile, Segment } from "../predictor-ui";
import type { FixtureSummary } from "../types";

/** A probability is only a probability if it is one. A missing field is not 0. */
function prob(x: number | null | undefined): number | null {
  return typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1 ? x : null;
}

function num(x: number | null | undefined): number | null {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

/** The three outcomes, in the order a fixture is read. The draw is its own
 *  segment: a three-way market rendered as two ways is a lie by layout. */
const SIDES = [
  { key: "home_win", label: "home" },
  { key: "draw", label: "draw" },
  { key: "away_win", label: "away" },
] as const;

export function panelFacts(fixture: FixtureSummary | null | undefined) {
  const tiles: MarketTile[] = [];
  const segments: Segment[] = [];
  const legend: Segment[] = [];
  if (!fixture) return { tiles, segments, legend };

  const edges = SIDES.map(({ key, label }) => ({
    key,
    label,
    edge: fixture[key],
    prob: prob(fixture[key]?.prob),
    implied: num(fixture[key]?.implied),
  }));
  const known = edges.filter((e): e is typeof e & { prob: number } => e.prob !== null);

  if (known.length) {
    // Labels are the teams where we have them, and the bare side otherwise —
    // "Arsenal 48%" is what a reader wants; "home 48%" is what we can always say.
    for (const e of known) {
      const label =
        e.key === "home_win" ? fixture.team_home : e.key === "away_win" ? fixture.team_away : "Draw";
      segments.push({ label, prob: e.prob, market: "result" });
      if (e.implied !== null) legend.push({ label, prob: e.implied, market: "result" });
    }

    // The tile carries the LEADING outcome, because a tile showing the draw's
    // 26% answers a question nobody asked; the bar beneath carries all three.
    const lead = [...known].sort((a, b) => b.prob - a.prob)[0];
    const leadLabel =
      lead.key === "home_win" ? fixture.team_home : lead.key === "away_win" ? fixture.team_away : "Draw";
    tiles.push({ market: "result", label: "result", value: pct(lead.prob), sub: `win · ${leadLabel}` });
  }

  const total = num(fixture.predicted_total_goals);
  if (total !== null) {
    tiles.push({ market: "total_goals", label: "total goals", value: stat(total) });
  }

  const btts = prob(fixture.btts_yes_prob);
  if (btts !== null) {
    tiles.push({ market: "btts", label: "both score", value: pct(btts) });
  }

  return { tiles, segments, legend };
}
