// Short explanations for the jargon-heavy stats shown around the app.
export const GLOSSARY = {
  edge:
    "The gap between the model's probability and the bookmaker's (de-vigged) implied probability. A positive edge means the model thinks that outcome is more likely than the market's price suggests.",
  marketImplied:
    "The bookmaker's own probability for this outcome, backed out from their odds and adjusted (de-vigged) to remove their built-in margin.",
  btts: "Both Teams To Score — the model's probability that both sides find the net at least once.",
  expectedCount:
    "The model's predicted average (λ) for this match — e.g. an expected value of 10.6 corners means the model's best guess is ~10-11 corners, priced against the line shown.",
  noLiveMarket:
    "There's no live bookmaker line for this market yet — The Odds API's free tier only covers match result and total goals in bulk. This is the model's own prediction with nothing to compare it against.",
  dataConfidenceNew:
    "One side has no recent match history the model has seen (e.g. just promoted after a long absence). The prediction still uses that team's live rating (which starts at a league-average default and updates after every match), it's just resting on little-to-no real data yet — expect it to sharpen quickly as their season gets underway.",
  dataConfidenceLimited:
    "One side has only a handful of matches of recent history to go on so far, so the prediction is still partly leaning on a league-average blend rather than fully trusting their specific recent form. It'll keep converging toward their real current form as more matches are played.",
  rps: "Ranked Probability Score — measures how good the predicted Win/Draw/Loss probabilities were, and (unlike plain accuracy) gives partial credit for near-misses since a draw is 'closer' to a home win than an away win is. Lower is better; 0 is a perfect forecast.",
  brier:
    "Brier score — the average squared error between predicted probabilities and the actual outcome. Lower is better; 0 is a perfect forecast.",
  dispersion:
    "Variance divided by mean of the actual counts. Close to 1 means a Poisson distribution fits well; noticeably above 1 (common for cards, which spike on red cards/melees) means the model switches to a negative binomial when pricing over/under lines.",
  mae: "Mean Absolute Error — on average, how many corners/cards off the model's prediction was, in either direction.",
  roi: "Return on Investment from the backtest's bets, as a percentage of the starting bankroll. A strongly positive number here usually signals overfitting rather than a real edge.",
  valueBetThreshold:
    "A fixture is flagged as a value bet when the model's probability beats the live bookmaker's implied probability by more than 5 percentage points.",
  scorelineGrid:
    "The full probability for every exact scoreline up to 5-5, from the same statistical model (Dixon-Coles or Bivariate-Poisson) that produces the win/draw/loss and goals predictions — darker cells are more likely.",
  recentForm:
    "This team's last 5 Premier League results, most recent on the right — win/draw/loss, not adjusted for opponent strength.",
  topScoreline:
    "The single most probable exact scoreline from the model's full grid — usually still well under 50% likely on its own, since goals are spread across many plausible scores.",
  anytimeScorer:
    "Probability this player scores at any point in the match, from their own recent goals-per-90-minutes rate — scaled up or down by how many goals the match model expects their team to score in this specific fixture, and by their typical playing time. No official betting line to compare against yet.",
  anytimeAssist: "Same idea as anytime goalscorer, but for assists — probability of setting up at least one goal.",
  playerAvailability:
    "Live status from the official FPL API. A doubtful player's probability is already discounted by their listed chance of playing; an injured/suspended player's probability is zeroed out. There's no live check for a very recent, not-yet-reflected change — a player who just picked up an injury could take a match or two to fully drop out of their own recent-form numbers.",
  powerRankings:
    "The scoreline model's own fitted attack and defence strength per team — not an opinion layered on top, but literally the numbers it uses internally to price every match. Net strength = attack minus defence. Higher attack = more goals scored; more negative defence = fewer goals conceded (it's a suppression term in the model, so more negative is better defensively).",
  ratingsTrend:
    "Elo and Pi ratings, replayed match-by-match through this season so far — two independent, simpler ways of tracking form over time (both reward wins and account for opponent strength) alongside the main model's own attack/defence numbers.",
  projectedTable:
    "Current actual points plus each team's *expected* points (3×P(win) + 1×P(draw), summed) from the scoreline model over every fixture left in the season — not a single simulated run, but the mathematical average across all the ways the rest of the season could go.",
  trackRecordScore:
    "RPS (match result) and Brier score (goals/BTTS) computed only on predictions that were actually logged before kickoff and have since been resolved against the real result — not a backtest on old seasons, a live, ongoing record starting from when this feature shipped.",
  biggestMisses:
    "Resolved predictions ranked by how far off they were (squared error) — the fixtures/markets where the model was most confidently wrong, worth a manual look for patterns (a particular market, a particular kind of team) worth improving.",
};
