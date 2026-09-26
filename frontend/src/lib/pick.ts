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
