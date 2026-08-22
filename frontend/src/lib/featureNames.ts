// Most feature columns follow {side}_{split?}_last_{window}_{stat} (e.g.
// "away_home_last_5_cards_for" = the away team's cards-for rate, from
// their last 5 *home* matches specifically). Parse that pattern into a
// readable label; fall back to simple cleanup for the rest (elo_diff,
// rest_days_home, h2h_home_win_rate, ...).
const ROLLING_RE = /^(home|away)_(?:(home|away)_)?last_(\d+)_(.+)$/;

export function formatFeatureName(raw: string): string {
  const m = raw.match(ROLLING_RE);
  if (m) {
    const [, side, split, window, stat] = m;
    const sideLabel = side === "home" ? "Home team" : "Away team";
    const splitLabel = split ? ` (${split} matches)` : "";
    const statLabel = stat.replace(/_/g, " ");
    return `${sideLabel}${splitLabel}: last ${window} ${statLabel}`;
  }
  return raw.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
