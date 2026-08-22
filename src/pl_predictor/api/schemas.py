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
    value_bet_flags: list[str]
    has_live_odds: bool


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
    status: str
    news: str
    confidence: str


class FixtureDetail(FixtureSummary):
    score_grid: list[list[float]]
    top_scorelines: list[dict]
    corners: OverUnderPrediction
    cards: OverUnderPrediction
    head_to_head: list[H2HMeeting]
    home_recent_form: list[str]
    away_recent_form: list[str]


class FixturePlayers(BaseModel):
    home_players: list[PlayerPrediction]
    away_players: list[PlayerPrediction]
