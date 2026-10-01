export type PickSide = "home_win" | "draw" | "away_win";

// The one pick a finished card is judged on: the most likely of home/draw/
// away. Mirrors tracking/store.py::_fixture_hit_table, whose Python max()
// keeps the first of equal values in home, draw, away order, so the card's
// verdict and the API's `hit` can never disagree.
export function matchPick(home: number, draw: number, away: number, teamHome: string, teamAway: string): { side: PickSide; label: string; prob: number } {
  let side: PickSide = "home_win";
  let prob = home;
  if (draw > prob) { side = "draw"; prob = draw; }
  if (away > prob) { side = "away_win"; prob = away; }
  const label = side === "home_win" ? `${teamHome} win` : side === "away_win" ? `${teamAway} win` : "Draw";
  return { side, label, prob };
}

/** The side a fixture's stored review judged, or null when there is no review.
 *  Read from the review's own "Match result" row — the row the finished card,
 *  the list card and the track record all agree with, because all three judge
 *  the plain argmax that `matchPick` above computes. */
export function reviewSide(postMatch: { verdicts?: { label: string; prediction: string; hit?: boolean }[] } | null | undefined): { side: PickSide; hit?: boolean } | null {
  const row = postMatch?.verdicts?.find((v) => v.label === "Match result");
  const prediction = row?.prediction;
  if (prediction !== "home_win" && prediction !== "draw" && prediction !== "away_win") return null;
  return typeof row?.hit === "boolean" ? { side: prediction, hit: row.hit } : { side: prediction };
}

/** Which side this fixture's pick IS, and the label the bar's segments use.
 *
 *  **A reviewed fixture's pick is the review's side**, and `predicted_result` is
 *  only the fallback for a fixture with no review at all. The two are not the
 *  same field: `predicted_result` is the DISPLAY variant, where the scoreline
 *  model promotes a draw when it and the percentage model agree one is likely,
 *  and the backend labels it informational only — never used to score accuracy
 *  (`api/schemas.py::_fill_predicted_result`). Measured on the shipped snapshot
 *  the display variant disagrees with the review's side on 17 of the 20 finished
 *  fixtures whose shots fields are present, and on 229 of the 330 unreviewed
 *  rows the review cannot contradict because there is no review.
 *
 *  So the review is read FIRST: it is the pick the review row, the list card and
 *  the track record all mean, and a block that named the other one would put three
 *  different picks on one page. It is also the ONLY source for the 30 finished
 *  fixtures baked into the public snapshot without `predicted_result` at all, so
 *  dropping it in favour of the display variant would leave those rows with no
 *  pick to state.
 *
 *  `label` is the SEGMENTS' vocabulary, not prose: the shared `panelFacts`
 *  adapter labels the three segments `team_home`, `team_away` and `"Draw"`, and
 *  the bar joins the pick to a segment by exact label match. A label worded any
 *  other way — `"Arsenal win"`, which is what `matchPick` returns for the card —
 *  joins to nothing and renders a bar with no accent at all. A pick is returned
 *  only when its side actually carries a probability, so a side with no
 *  probability names no pick and there is no segment for one to fail to find.
 *
 *  Returned as one value rather than read by the caller, because the rule has
 *  three sources and two fallbacks and a caller re-deriving any part of it is how
 *  this went wrong the first time. */
export function resolvedPick(detail: {
  team_home: string;
  team_away: string;
  home_win?: { prob?: number | null } | null;
  draw?: { prob?: number | null } | null;
  away_win?: { prob?: number | null } | null;
  predicted_result?: string | null;
  post_match?: { verdicts?: { label: string; prediction: string; hit?: boolean }[] } | null;
}): { side: PickSide; label: string; prob: number; was_right?: boolean } | undefined {
  const review = reviewSide(detail.post_match);
  const key = review?.side ?? (["home_win", "draw", "away_win"] as const).find((k) => detail.predicted_result === k) ?? null;
  const prob = key ? detail[key]?.prob : undefined;
  if (!key || typeof prob !== "number" || !Number.isFinite(prob)) return undefined;
  return {
    side: key,
    label: key === "home_win" ? detail.team_home : key === "away_win" ? detail.team_away : "Draw",
    prob,
    // The review's `hit` flag, attached only when the pick IS the review's own
    // side. It is not about the display variant's side, so it is dropped rather
    // than hung on a pick it does not describe.
    ...(review && review.side === key && typeof review.hit === "boolean" ? { was_right: review.hit } : {}),
  };
}
