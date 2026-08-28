"""schemas.py — Pydantic response models for the fixture endpoints (the
primary UI surface, worth a real typed contract). The metadata/eval
endpoints (manifest, calibration, backtest) return plain dicts — those
mirror `models/manifest.py`'s manifest.json / `evaluate/*`'s return dicts
closely enough that a parallel Pydantic model would just be duplication.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class MarketEdge(BaseModel):
    prob: float
    implied: float | None = None
    edge: float | None = None


class SingleBetRecommendation(BaseModel):
    market: str
    probability: float
    implied_probability: float
    edge: float
    price: float
    bookmaker: str


class FixtureSummary(BaseModel):
    event_id: str
    commence_time: datetime
    team_home: str
    team_away: str
    home_win: MarketEdge
    draw: MarketEdge
    away_win: MarketEdge
    over_2_5: MarketEdge
    under_2_5: MarketEdge
    btts_yes_prob: float
    top_scoreline: str
    is_fallback_prediction: bool
    data_confidence: str | None = None
    # Derived from the scoreline model's own home/away goal expectations
    # (sum for total goals, difference for margin/"spread") rather than a
    # separate model — measured directly to be the more accurate choice;
    # see value_bets.py::build_value_bet_table's comment for the numbers.
    # None for already-finished fixtures reached via a historical tracking
    # record that predates this field.
    predicted_total_goals: float | None = None
    predicted_margin: float | None = None
    value_bet_flags: list[str]
    has_live_odds: bool
    odds_fetched_at: datetime | None = None
    odds_is_stale: bool = False
    recommended_bet: SingleBetRecommendation | None = None


class OverUnderPrediction(BaseModel):
    lambda_: float
    line: float
    over: float
    under: float


class H2HMeeting(BaseModel):
    date: str
    team_home: str
    team_away: str
    goals_home: int
    goals_away: int


class PlayerPrediction(BaseModel):
    player_id: int
    name: str
    position: str
    anytime_goal_prob: float
    anytime_assist_prob: float
    anytime_goal_contribution_prob: float
    status: str
    news: str
    confidence: str
    predicted_starter: bool
    confirmed_starter: bool
    expected_minutes: float
    is_penalty_taker: bool
    is_set_piece_taker: bool


class FixtureTeamContext(BaseModel):
    rest_days: int | None = None
    xg_for_last_5: float | None = None
    xg_against_last_5: float | None = None
    corners_last_5: float | None = None
    cards_last_5: float | None = None
    set_piece_xg_share_last_5: float | None = None


class FixtureActualStats(BaseModel):
    home: dict[str, float | int | None]
    away: dict[str, float | int | None]


class FixtureDetail(FixtureSummary):
    score_grid: list[list[float]]
    top_scorelines: list[dict]
    corners: OverUnderPrediction
    cards: OverUnderPrediction
    head_to_head: list[H2HMeeting]
    home_recent_form: list[str]
    away_recent_form: list[str]
    home_context: FixtureTeamContext
    away_context: FixtureTeamContext
    post_match: dict | None = None
    actual_stats: FixtureActualStats | None = None


class FixturePlayers(BaseModel):
    home_players: list[PlayerPrediction]
    away_players: list[PlayerPrediction]
