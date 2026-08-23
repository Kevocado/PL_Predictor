export interface MarketEdge {
  prob: number;
  implied: number | null;
  edge: number | null;
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
  value_bet_flags: string[];
  has_live_odds: boolean;
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
}

export interface PlayerPrediction {
  player_id: number;
  name: string;
  position: string;
  anytime_goal_prob: number;
  anytime_assist_prob: number;
  status: string;
  news: string;
  confidence: string;
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
  confidence: "new" | "limited" | "established";
  fitted: boolean;
}

export interface RatingPoint {
  date: string;
  elo: number;
  pi: number;
}

export interface RankingsResponse {
  rankings: TeamRanking[];
  ratings_history: Record<string, RatingPoint[]>;
  season: string;
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
