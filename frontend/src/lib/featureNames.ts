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

const STAT_EXPLANATIONS: Record<string, string> = {
  goals_for: "goals scored",
  goals_against: "goals conceded",
  shots_for: "shots taken",
  shots_against: "shots faced",
  shots_on_target_for: "shots on target taken",
  shots_on_target_against: "shots on target faced",
  corners_for: "corners won",
  corners_against: "corners conceded",
  cards_for: "cards (yellow + red) received",
  cards_against: "cards the opponent received",
  points: "league points earned (3 for a win, 1 for a draw, 0 for a loss)",
};

/** A longer, plain-English explanation of what a raw feature column
 * actually measures — for hover tooltips, not for axis labels (that's
 * `formatFeatureName`). Covers every pattern the feature pipeline
 * currently produces (features/build.py, features/rolling_form.py,
 * features/xg_form.py, features/ratings.py, features/referee.py,
 * features/head_to_head.py, features/rest_days.py); falls back to a
 * generic message for anything new added later that this hasn't been
 * updated to cover yet. */
export function explainFeatureName(raw: string): string {
  const rolling = raw.match(ROLLING_RE);
  if (rolling) {
    const [, side, split, window, stat] = rolling;
    const sideLabel = side === "home" ? "the home team" : "the away team";
    const statLabel = STAT_EXPLANATIONS[stat] ?? stat.replace(/_/g, " ");
    const venueNote = split ? ` in ${split} matches only` : " regardless of venue";
    return `Average ${statLabel} by ${sideLabel} per match over their last ${window} games${venueNote}, using only matches strictly before this one.`;
  }

  const xg = raw.match(/^(home|away)_xg_(for|against)_last_(\d+)$/);
  if (xg) {
    const [, side, direction, window] = xg;
    const sideLabel = side === "home" ? "the home team" : "the away team";
    const dirLabel = direction === "for" ? "generated" : "conceded";
    return `Average expected goals (xG) ${dirLabel} by ${sideLabel} per match over their last ${window} games — a shot-quality-based measure of scoring chances, not just the actual goals scored.`;
  }

  const xgDelta = raw.match(/^(home|away)_xg_delta_(for|against)_last_(\d+)$/);
  if (xgDelta) {
    const [, side, direction, window] = xgDelta;
    const sideLabel = side === "home" ? "The home team's" : "The away team's";
    const dirLabel = direction === "for" ? "goals scored vs. their own underlying xG" : "goals conceded vs. their opponents' underlying xG";
    return `${sideLabel} recent ${dirLabel}, over their last ${window} games. Positive means they've outperformed their chances lately (often finishing well, or a bit lucky) — a signal that tends to regress toward the mean.`;
  }

  const fixed: Record<string, string> = {
    elo_home: "The home team's Elo rating right before this match — a single number summarizing overall strength, updated after every result.",
    elo_away: "The away team's Elo rating right before this match.",
    elo_diff: "Home team's Elo rating minus the away team's — positive means the home side is rated stronger overall.",
    pi_home: "The home team's Pi rating right before this match — an alternative strength rating that reacts faster to recent goal difference than Elo does.",
    pi_away: "The away team's Pi rating right before this match.",
    pi_diff: "Home team's Pi rating minus the away team's — positive means the home side is rated stronger.",
    h2h_home_goal_diff_avg: "Average goal difference (home minus away) across their last few head-to-head meetings.",
    h2h_home_win_rate: "How often the home team has won their last few meetings against this specific opponent.",
    rest_days_home: "Days since the home team's last match — fewer days can mean fatigue.",
    rest_days_away: "Days since the away team's last match.",
    is_first_match_of_season_home: "Whether this is the home team's first match of the season — no recent form to draw on yet.",
    is_first_match_of_season_away: "Whether this is the away team's first match of the season.",
    referee_card_rate: "This referee's average total cards shown per match they've officiated recently — some referees are simply stricter than others.",
  };
  return fixed[raw] ?? "No further detail available for this feature yet.";
}
