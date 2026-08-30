export interface MarketEdge {
  prob: number;
  implied: number | null;
  edge: number | null;
}

export interface SingleBetRecommendation {
  market: string;
  probability: number;
  implied_probability: number;
  edge: number;
  price: number;
  bookmaker: string;
}

export interface FixtureSummary {
  event_id: string;
  commence_time: string;
  team_home: string;
  team_away: string;
  home_win: MarketEdge;
  draw: MarketEdge;
  away_win: MarketEdge;
  over_2_5: MarketEdge;
  under_2_5: MarketEdge;
  btts_yes_prob: number;
  top_scoreline: string;
  is_fallback_prediction: boolean;
  data_confidence: "new" | "limited" | "established" | null;
  // Derived from the scoreline model's own home/away goal expectations
  // (sum for total goals, difference for margin) rather than a separate
  // model. null for already-finished fixtures from an older tracking
  // record that predates this field.
  predicted_total_goals: number | null;
  predicted_margin: number | null;
  // P(team scores >=2), same derived-from-the-grid reasoning and the same
  // null-for-pre-existing-tracked-records caveat as the two fields above.
  home_2plus_prob: number | null;
  away_2plus_prob: number | null;
  value_bet_flags: string[];
  has_live_odds: boolean;
  odds_fetched_at: string | null;
  odds_is_stale: boolean;
  recommended_bet: SingleBetRecommendation | null;
}

export interface CurrentGameweekFixture {
  event_id: string;
  team_home: string;
  team_away: string;
  commence_time: string;
  finished: boolean;
  actual_goals_home: number | null;
  actual_goals_away: number | null;
  predicted_home_win: number;
  predicted_draw: number;
  predicted_away_win: number;
  predicted_scoreline: string | null;
  hit: boolean | null;
  backfilled: boolean;
  has_live_odds: boolean;
  value_bet_flags: string[];
  home_player_events?: FixturePlayerEvent[];
  away_player_events?: FixturePlayerEvent[];
  player_events_pending?: boolean;
}

export interface FixturePlayerEvent {
  name: string;
  goals: number;
  assists: number;
}

export interface CurrentGameweekResponse {
  gameweek: number | null;
  fixtures: CurrentGameweekFixture[];
  is_current: boolean;
  min_gameweek: number | null;
  max_gameweek: number | null;
}

export interface OverUnderPrediction {
  lambda_: number;
  line: number;
  over: number;
  under: number;
}

export interface H2HMeeting {
  date: string;
  team_home: string;
  team_away: string;
  goals_home: number;
  goals_away: number;
}

export interface ScorelineEntry {
  home: number;
  away: number;
  prob: number;
}

export interface FixtureDetail extends FixtureSummary {
  score_grid: number[][];
  top_scorelines: ScorelineEntry[];
  corners: OverUnderPrediction;
  cards: OverUnderPrediction;
  head_to_head: H2HMeeting[];
  home_recent_form: string[];
  away_recent_form: string[];
  home_context: FixtureTeamContext;
  away_context: FixtureTeamContext;
  post_match: FixturePostMatch | null;
  actual_stats: FixtureActualStats | null;
  pre_match_value_bets: FixtureValueBetSnapshot[];
}

export interface FixtureValueBetSnapshot {
  market: string;
  probability: number;
  implied_probability: number;
  edge: number;
  price: number;
  bookmaker: string | null;
  snapshotted_at: string;
  resolved: boolean;
  won: boolean | null;
  final_score: string | null;
  result_source: string | null;
}

export interface PostMatchVerdict {
  label: string;
  prediction: string;
  actual: string;
  hit: boolean;
}

export interface PostMatchPlayerCall {
  name: string;
  team: string;
  goal_probability: number;
  assist_probability: number;
  contribution_probability: number;
  goals: number;
  assists: number;
  goal_hit: boolean;
  assist_hit: boolean;
  goal_called: boolean;
  assist_called: boolean;
  contribution_called: boolean;
  is_recommended: boolean;
  contribution_hit: boolean;
  provenance: "snapshot" | "reconstructed";
}

export interface FixturePostMatch {
  final_score: string;
  provenance: "snapshot" | "reconstructed";
  verdicts: PostMatchVerdict[];
  player_calls: PostMatchPlayerCall[];
}

export interface FixturePlayerReviewCall {
  name: string;
  team: string;
  goal_probability: number;
  assist_probability: number;
  contribution_probability: number;
  goals: number;
  assists: number;
  hit: boolean;
  is_recommended: boolean;
  review_label: "Goal call" | "Assist call" | "G+A call" | "Recommended player" | "Overperformer";
  review_probability: number;
  review_market: "Goal" | "Assist" | "G+A";
}

export interface FixturePlayerReview {
  provenance: "snapshot" | "reconstructed";
  correct: FixturePlayerReviewCall[];
  missed: FixturePlayerReviewCall[];
  overperformed: FixturePlayerReviewCall[];
}

export interface FixtureTeamContext {
  rest_days: number | null;
  xg_for_last_5: number | null;
  xg_against_last_5: number | null;
  corners_last_5: number | null;
  cards_last_5: number | null;
  set_piece_xg_share_last_5: number | null;
}

export interface FixtureActualStats {
  home: Record<string, number | null>;
  away: Record<string, number | null>;
}

export interface PlayerPrediction {
  player_id: number;
  name: string;
  position: string;
  anytime_goal_prob: number;
  anytime_assist_prob: number;
  anytime_goal_contribution_prob: number;
  status: string;
  news: string;
  confidence: string;
  predicted_starter: boolean;
  confirmed_starter: boolean;
  expected_minutes: number;
  is_penalty_taker: boolean;
  is_set_piece_taker: boolean;
}

export interface FixturePlayers {
  home_players: PlayerPrediction[];
  away_players: PlayerPrediction[];
}

export interface CalibrationStat {
  rps: number;
  brier: number;
  ignorance?: number;
  n_matches: number;
}

export interface CalibrationResponse {
  model: CalibrationStat;
  bookmaker: CalibrationStat | null;
  naive: CalibrationStat;
  season: string | null;
}

export interface BacktestResults {
  "Total Bets": number;
  "Successful Bets": number;
  "Successful Bet %": number;
  "Max Bankroll": number | null;
  "Min Bankroll": number | null;
  Profit: number;
  ROI: number;
}

export interface BacktestResponse {
  results: BacktestResults;
  bankroll_curve: number[];
  staking: "kelly" | "flat";
  season: string | null;
  selections: HistoricalValueBet[];
}

export interface HistoricalValueBet {
  date: string;
  fixture: string;
  selection: string;
  price: number;
  model_probability: number;
  implied_probability: number;
  edge: number;
  won: boolean;
}

export interface BettingValidationSummary {
  bets: number;
  wins: number;
  win_rate: number | null;
  yield: number | null;
  yield_ci_95: [number, number] | null;
}

export interface BettingValidationBreakdown {
  label: string;
  bets: number;
  wins: number;
  win_rate: number | null;
  yield: number | null;
}

export interface WalkForwardBettingResponse {
  model: string;
  min_train_seasons: number;
  summary: BettingValidationSummary;
  folds: Array<BettingValidationBreakdown & { season: string; train_matches: number }>;
  by_market: BettingValidationBreakdown[];
  by_odds_band: BettingValidationBreakdown[];
}

export interface ValueBetTrackRecordResponse {
  n_flagged: number;
  n_resolved: number;
  n_pending: number;
  confirmed_wins: number;
  confirmed_losses: number;
  confirmed_win_rate: number | null;
  confirmed_bets: ConfirmedValueBet[];
  results: BacktestResults | null;
  bankroll_curve: number[];
  staking: "kelly" | "flat";
}

export interface ConfirmedValueBet {
  fixture: string;
  market: string;
  selection: string;
  price: number;
  edge: number;
  won: boolean;
  result_source: string;
  resolved_at: string | null;
}

export interface FeatureImportance {
  gain: Record<string, number>;
  permutation: Record<string, number>;
  shap: Record<string, number>;
}

export interface ManifestModelMetrics {
  metrics: Record<string, number>;
  dispersion?: number;
  path?: string;
  importance?: FeatureImportance;
}

export interface TeamRanking {
  team: string;
  attack: number;
  defence: number;
  net_strength: number;
  games_played: number | null;
  confidence: "new" | "limited" | "preseason" | "established";
  fitted: boolean;
}

export interface RatingPoint {
  date: string;
  elo: number;
  pi: number;
  gameweek: number | null;
}

export interface RankingsResponse {
  rankings: TeamRanking[];
  ratings_history: Record<string, RatingPoint[]>;
  season: string;
}

export interface TeamHubRecentMatch {
  date: string;
  opponent: string;
  venue: "Home" | "Away";
  result: "W" | "D" | "L";
  score: string;
  xg_for: number | null;
  xg_against: number | null;
}

export interface TeamHubTeam {
  team: string;
  played: number;
  points: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
  assists: number;
  xa: number | null;
  points_per_match: number | null;
  form_points_per_match: number | null;
  form_trend: "up" | "down" | "steady" | "new";
  goals_for_per_match: number | null;
  goals_against_per_match: number | null;
  shots_per_match: number | null;
  shots_on_target_per_match: number | null;
  corners_per_match: number | null;
  fouls_per_match: number | null;
  cards_per_match: number | null;
  xg_for: number | null;
  xg_against: number | null;
  goals_minus_xg: number | null;
  goals_conceded_minus_xg: number | null;
  set_piece_xg_share: number | null;
  streak: number;
  recent_matches: TeamHubRecentMatch[];
}

export interface TeamHubResponse {
  season: string;
  teams: TeamHubTeam[];
}

export interface PlayerHubPlayer {
  id: number;
  name: string;
  team: string;
  position: string;
  status: string;
  chance_of_playing: number | null;
  minutes: number;
  starts: number;
  goals: number;
  assists: number;
  xg: number | null;
  xa: number | null;
  xgi: number | null;
  threat: number | null;
  creativity: number | null;
  ict: number | null;
  bps: number;
  bonus: number;
  news: string;
  quality_rating: number | null;
  form_rating: number;
  live_form_rating: number;
  live_form_vs_quality: number | null;
  overall_rating: number | null;
  current_impact_rating: number;
  rating_driver: string;
  rating_expected_minutes: number;
  rating_status: "established" | "provisional";
  rating_evidence_minutes: number;
  rating_model_source: string;
}

export interface PlayerHubResponse {
  players: PlayerHubPlayer[];
  leaderboards: Record<string, PlayerHubPlayer[]>;
  rating_model_source: string;
  data_freshness: string;
}

export interface ScorerAccuracyGroup {
  calls: number;
  call_hits: number;
  call_hit_rate: number | null;
  goal_brier: number | null;
  calibration: Array<{ range: string; n: number; predicted: number; actual: number }>;
}

export interface ScorerAccuracyResponse {
  snapshot: ScorerAccuracyGroup;
  reconstructed: ScorerAccuracyGroup;
}

export interface ProjectedTableRow {
  team: string;
  played: number;
  current_points: number;
  current_position: number | null;
  projected_points: number;
  projected_goal_diff: number;
  projected_position: number;
  position_delta: number | null;
}

export interface ProjectedTableResponse {
  table: ProjectedTableRow[];
  season: string;
}

export interface GameweekTrendPoint {
  gameweek: number;
  pct_correct: number;
  n_fixtures: number;
}

export interface TrackRecordSummary {
  n_resolved_fixtures: number;
  pct_correct_overall: number | null;
  current_gameweek: number | null;
  pct_correct_current_gameweek: number | null;
  n_fixtures_current_gameweek: number;
  gameweek_trend: GameweekTrendPoint[];
}

export interface BiggestUpset {
  team_home: string;
  team_away: string;
  commence_time: string;
  gameweek: number | null;
  actual_goals_home: number | null;
  actual_goals_away: number | null;
  actual_outcome: "home_win" | "draw" | "away_win";
  predicted_prob: number;
}

export interface GameweekResult {
  event_id: string;
  team_home: string;
  team_away: string;
  commence_time: string;
  predicted_scoreline: string | null;
  actual_goals_home: number | null;
  actual_goals_away: number | null;
  predicted_home_win: number;
  predicted_draw: number;
  predicted_away_win: number;
  actual_outcome: "home_win" | "draw" | "away_win" | null;
  hit: boolean;
  backfilled: boolean;
}

export interface GameweekGroup {
  gameweek: number | null;
  pct_correct: number;
  n_fixtures: number;
  fixtures: GameweekResult[];
}

export interface TrackRecordResponse {
  summary: TrackRecordSummary;
  biggest_upsets: BiggestUpset[];
  gameweeks: GameweekGroup[];
}

export interface ManifestResponse {
  trained_at: string;
  seasons: string[];
  n_train: number;
  n_val: number;
  n_current_season_matches: number;
  live_current_season_matches?: number;
  live_results_source?: string;
  features: string[];
  scoreline: {
    chosen_model: string;
    dixon_coles: ManifestModelMetrics;
    bivariate_poisson: ManifestModelMetrics;
    ml_scoreline: ManifestModelMetrics & {
      teams?: string[];
      importance_home?: FeatureImportance;
      importance_away?: FeatureImportance;
    };
  };
  corners: ManifestModelMetrics;
  cards: ManifestModelMetrics;
}

export interface ManifestHistoryEntry {
  trained_at: string;
  n_train: number;
  n_current_season_matches: number;
  chosen_model: string;
  rps: number;
  brier: number;
}

export interface ManifestHistoryResponse {
  history: ManifestHistoryEntry[];
}

export interface FPLFixtureProjection {
  opponent: string;
  was_home: boolean;
  difficulty: number;
  expected_goals: number;
  clean_sheet_probability: number;
  expected_minutes: number;
  model_source: string;
}

export interface FPLPlayerProjection {
  player_id: number;
  name: string;
  web_name: string;
  team: string;
  team_id: number;
  position: "GK" | "DEF" | "MID" | "FWD";
  price: number;
  status: string;
  news: string;
  availability: number;
  expected_minutes: number;
  projected_points: number;
  fixture_count: number;
  fixtures: FPLFixtureProjection[];
  drivers: string[];
}

export interface FPLProjectionsResponse {
  gameweek: number;
  generated_at: string;
  model_source: string;
  data_freshness: string;
  players: FPLPlayerProjection[];
}

export interface FPLRecommendation {
  starting_xi: FPLPlayerProjection[];
  captain: FPLPlayerProjection | null;
  vice_captain: FPLPlayerProjection | null;
  bench: FPLPlayerProjection[];
  projected_points: number;
}

export interface FPLSquadResponse extends FPLRecommendation {
  gameweek: number;
  model_source: string;
  squad: FPLPlayerProjection[];
  budget: number;
  spent: number;
  remaining: number;
}

export interface FPLTransferIdea {
  out: FPLPlayerProjection;
  in: FPLPlayerProjection;
  cost: number;
  projected_gain: number;
  net_gain: number;
}

export interface FPLCurrentLineup extends FPLRecommendation {
  squad: FPLPlayerProjection[];
}

export interface FPLTransfersResponse {
  gameweek: number;
  source_gameweek?: number;
  source: "manual" | "public_entry";
  entry_id?: number;
  free_transfers: number;
  bank: number;
  current_lineup: FPLCurrentLineup;
  recommendations: FPLTransferIdea[];
}
