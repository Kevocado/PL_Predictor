/** The pick, as the SHIPPED data path builds it — no mocks, no fixtures.
 *
 *  The mock-based tests prove the modal picks a side when a detail is handed to
 *  it. This one reads `data/public_snapshot.json` — the file the site actually
 *  serves — and proves the same rule holds for every one of the 380 rows in it,
 *  including the 30 that carry no `predicted_result` at all.
 *
 *  The rule: **a reviewed fixture's pick is the review's own side.** The pick
 *  comes from the stored "Match result" verdict, and only falls back to
 *  `predicted_result` on a fixture with no review. That matches
 *  `lib/pick.ts`'s `matchPick` and tracking's `_fixture_hit_table`, which both
 *  judge the plain argmax.
 *
 *  It matters because `predicted_result` is the DISPLAY variant: the scoreline
 *  model promotes a draw when it and the percentage model agree one is likely,
 *  and the backend labels it informational only, never used to score accuracy
 *  (`api/schemas.py::_fill_predicted_result`). Reading it as the pick put a
 *  different side on the block than on the review row, the list card and the
 *  track record — four copies of one number, and three of them right.
 *
 *  No network and no api mock: this reads a committed file, so it cannot pass on
 *  a fixture that happens to agree.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { matchPick, resolvedPick } from "../lib/pick";

const snapshot = JSON.parse(
  readFileSync(resolve(__dirname, "../../../data/public_snapshot.json"), "utf8"),
) as { fixture_detail_by_event_id: Record<string, any> };

const details = Object.entries(snapshot.fixture_detail_by_event_id);

/** The plain argmax over the three sides, skipping a side with no probability.
 *  `matchPick` needs three numbers and this site may not have three, so a row
 *  with an absent side is not comparable and is excluded rather than zero-filled. */
const argmax = (d: any): "home_win" | "draw" | "away_win" | null => {
  let best: "home_win" | "draw" | "away_win" | null = null;
  let bestProb = -Infinity;
  for (const k of ["home_win", "draw", "away_win"] as const) {
    const p = d[k]?.prob;
    if (typeof p !== "number") continue;
    if (p > bestProb) {
      bestProb = p;
      best = k;
    }
  }
  return best;
};

/** The pick the shipped modal builds — the SHIPPED function, not a copy of it.
 *
 *  This was originally re-derived here, which made every assertion below
 *  unfalsifiable by the component: the copy agreed with the copy, so reverting
 *  the component's rule to `predicted_result`-first left this file green. The
 *  rule is therefore imported from `lib/pick`, the same module `FixtureModal`
 *  calls, so the assertions below are statements about what ships. */
const shippedPick = (d: any): string | null => resolvedPick(d)?.side ?? null;

describe("the pick, over the shipped snapshot itself", () => {
  it("has 380 fixture details to check", () => {
    // Guard rather than assumption: every count below is a claim about this file,
    // and a rename of the key would quietly make them all vacuously true.
    expect(details).toHaveLength(380);
  });

  it("names a pick for every renderable row, from the review or the display variant", () => {
    const unnamed = details.filter(([, d]) => d.post_match && shippedPick(d) === null);
    expect(unnamed.map(([id]) => id)).toEqual([]);
  });

  it("takes the review's side for every reviewed row, never the display variant", () => {
    const reviewed = details.filter(([, d]) => d.post_match);
    const mismatched = reviewed.filter(([, d]) => {
      const review = d.post_match.verdicts.find((v: any) => v.label === "Match result")?.prediction;
      return review && shippedPick(d) !== review;
    });
    expect(mismatched.map(([id]) => id)).toEqual([]);
  });

  it("agrees with the tracked argmax on every REVIEWED row, which is what the record judges", () => {
    // The independent check, and scoped to the reviewed rows on purpose.
    //
    // Measured on the shipped snapshot: the review's "Match result" verdict equals
    // the plain argmax on all 50 finished fixtures. That is the rule this rule set
    // exists to match — `matchPick` and `_fixture_hit_table` both judge the argmax,
    // so the review and the track record can never disagree.
    //
    // It is NOT asserted across all 380 rows, because that would be false and the
    // reason is the whole point of the two-source pick. On the 330 unreviewed rows
    // `predicted_result` is deliberately the scoreline model's draw-promoted
    // variant and disagrees with the argmax on 229 of them: that is the display
    // variant doing its job on a fixture with no review and therefore no record to
    // contradict. Reading the argmax there would make the block contradict the
    // scoreline model the rest of that row shows.
    const reviewed = details.filter(([, d]) => d.post_match);
    expect(reviewed).toHaveLength(50);

    const disagree = reviewed.filter(([, d]) => shippedPick(d) !== argmax(d));
    expect(disagree.map(([id]) => id)).toEqual([]);
    // And the same rows agree through the site's own helper, so the two
    // implementations of the argmax are the same rule rather than a coincidence.
    for (const [, d] of reviewed) {
      const m = matchPick(d.home_win.prob, d.draw.prob, d.away_win.prob, d.team_home, d.team_away);
      expect(shippedPick(d), d.event_id).toBe(m.side);
    }
  });

  it("still takes the display variant where there is no review to override it", () => {
    // The other half, and the control on the test above: if `shippedPick` read the
    // argmax everywhere, the 50 reviewed rows would agree for the wrong reason.
    // Measured on the snapshot: 330 unreviewed rows, of which 229 carry a
    // draw-promoted `predicted_result` the argmax does not share. Those rows must
    // name the model's displayed pick, or the block would contradict the scoreline
    // figures printed directly beneath it.
    const unreviewed = details.filter(([, d]) => !d.post_match);
    expect(unreviewed).toHaveLength(330);
    const drawPromoted = unreviewed.filter(([, d]) => d.predicted_result === "draw" && argmax(d) !== "draw");
    expect(drawPromoted.length).toBeGreaterThan(200);
    for (const [id, d] of drawPromoted) {
      expect(shippedPick(d), id).toBe("draw");
    }
  });

  it("still disagrees with the display variant on 17 of the 20 renderable finished rows", () => {
    // The measurement that makes the rule above worth anything. If `predicted_result`
    // and the tracked argmax had always agreed, preferring the review would be
    // free and this test would prove nothing. On the shipped snapshot the display
    // variant disagrees with the review's side on 17 of the 20 finished fixtures
    // whose shots fields are present — and on 17 of the 50 finished fixtures
    // overall, because the other 30 are exactly the rows carrying no
    // `predicted_result` at all.
    //
    // Pinned as a lower bound rather than an exact count on purpose: the snapshot
    // is refreshed by an automated job, so an exact figure here would turn a
    // routine data refresh into a failing test. The bound still fails if the two
    // collapse onto each other, which is the thing this is watching for.
    const renderable = details.filter(([, d]) => d.post_match && "home_shots" in d);
    const finished = details.filter(([, d]) => d.post_match);
    const differ = (rows: [string, any][]) =>
      rows.filter(([, d]) => {
        const review = d.post_match.verdicts.find((v: any) => v.label === "Match result")?.prediction;
        return review && d.predicted_result && d.predicted_result !== review;
      }).length;

    expect(renderable).toHaveLength(20);
    expect(finished).toHaveLength(50);
    expect(differ(renderable)).toBeGreaterThanOrEqual(15);
    expect(differ(finished)).toBeGreaterThanOrEqual(15);
  });

  it("still names a pick for the 30 rows that carry no predicted_result at all", () => {
    // These are the rows PL#34 is about: 30 backfilled finished fixtures with no
    // `predicted_result` key AND no shot projections. Each has a stored review,
    // so each must still name a pick — and each must also render without a
    // `NaN`, which the modal test covers against the `hasNumber` guards. The two
    // properties are separate and both are needed: a row that rendered cleanly
    // while naming no pick would be blank and honest at once.
    const noKey = details.filter(([, d]) => !("predicted_result" in d));
    expect(noKey).toHaveLength(30);
    expect(noKey.every(([, d]) => d.post_match)).toBe(true);
    expect(noKey.filter(([, d]) => shippedPick(d) === null).map(([id]) => id)).toEqual([]);

    // The same 30 are the rows missing the shots keys, so one assertion covers
    // both halves: this is the set that blanked the modal before PL#34.
    const noShots = details.filter(([, d]) => !("home_shots" in d));
    expect(noShots.map(([id]) => id).sort()).toEqual(noKey.map(([id]) => id).sort());
  });
});