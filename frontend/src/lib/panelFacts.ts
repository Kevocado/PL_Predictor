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
import type { MarketTile, PickRef, Segment } from "../predictor-ui";
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

/** What PL's `/facts` calls the two sides. `match_pick` in the PL API's facts
 *  module words the pick `"<team> win"` and the draw `"Draw"` — the draw needs no
 *  suffix, which is exactly why a draw pick has always matched this site's bar
 *  and a side pick never has. */
const WIN_SUFFIX = " win";

/** The answer's pick, in the vocabulary THIS site's bar is drawn in.
 *
 *  The bar joins the pick to a segment **by label** (`pickIndex` in
 *  `ProbabilityBar`, which is spec §5b: the service derives the pick and the
 *  renderer follows it). This site labels its segments with the bare team names,
 *  because "Arsenal 48%" is what a reader wants and "Arsenal win 48%" is not —
 *  and PL's facts say "Arsenal win". So a home or away pick matched nothing, and
 *  a bar with nothing matched is the panel's correct rendering of a bundle with
 *  **no pick** and a completely wrong one for a fixture that has one. It fails
 *  silently: the build is green, every test passes, and the reader is told which
 *  side was picked by a verdict sentence above a bar that declines to agree.
 *
 *  Only the STRING is translated. The pick's identity — which side, and at what
 *  probability — is the service's, derived server-side from the validated facts,
 *  and it is never re-derived here from this site's own numbers: a site that
 *  worked the pick out for itself would reintroduce the exact defect the field
 *  exists to remove, and would accent whichever segment happened to be widest.
 *
 *  A label that cannot be placed is returned **unchanged**, so the bar fails
 *  closed — nothing accented — rather than accenting the nearest segment to a
 *  claim nobody made.
 *
 *  Takes a pick rather than a pick-or-nothing on purpose. The contract states
 *  "no pick" by OMITTING the key and never as `null` ("`is there a pick` stays
 *  one question with one answer"), so whether there is a pick is the caller's
 *  question to ask before it gets here; this function only ever translates one
 *  that exists, and cannot turn a real pick into `null` on the way through.
 */
export function barPick(pick: PickRef, segments: Segment[]): PickRef {
  if (segments.some((s) => s.label === pick.label)) return pick;
  if (!pick.label.endsWith(WIN_SUFFIX)) return pick;
  const bare = pick.label.slice(0, -WIN_SUFFIX.length);
  const match = segments.find((s) => s.label === bare);
  return match ? { ...pick, label: match.label } : pick;
}
