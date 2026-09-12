"""schemas.py — Pydantic response models for the fixture endpoints (the
primary UI surface, worth a real typed contract). The metadata/eval
endpoints (manifest, calibration, backtest) return plain dicts — those
mirror `models/manifest.py`'s manifest.json / `evaluate/*`'s return dicts
closely enough that a parallel Pydantic model would just be duplication.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator

from ..outcomes import predicted_result


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
    # Both computed from top_scoreline + the three MarketEdge probs above
    # (see outcomes.py) rather than set by callers -- predicted_result is
    # the marginal argmax promoted to "draw" when the scoreline model and
    # the percentage model agree one is likely; draw_signal is just
    # predicted_result == "draw", broken out as its own bool so the
    # frontend can badge a draw call ("model leans draw") without string
    # comparison. Informational only -- NOT used for accuracy/hit
    # determination (tracking/store.py uses plain marginal argmax there;
    # a walk-forward backtest showed this promotion is net-negative for
    # raw accuracy, see store.py::get_fixture_post_match).
    predicted_result: str = ""
    draw_signal: bool = False
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
    # P(team scores >=2), same "derived from the grid, not a separate
    # model" reasoning, and the same None-for-pre-existing-tracked-records
    # backward-compatibility caveat as predicted_total_goals/margin above.
    home_2plus_prob: float | None = None
    away_2plus_prob: float | None = None
    value_bet_flags: list[str]
    has_live_odds: bool
    odds_fetched_at: datetime | None = None
    odds_is_stale: bool = False
    recommended_bet: SingleBetRecommendation | None = None

    @model_validator(mode="after")
    def _fill_predicted_result(self) -> "FixtureSummary":
        self.predicted_result = predicted_result(self.top_scoreline, self.home_win.prob, self.draw.prob, self.away_win.prob)
        self.draw_signal = self.predicted_result == "draw"
        return self


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
    # None when the Understat crosswalk has no match for this player (or no
    # shots data exists yet) -- never a fabricated number. See
    # docs/superpowers/specs/2026-09-04-player-shots-market-design.md.
    expected_shots: float | None = None
    expected_shots_on_target: float | None = None
    anytime_shot_on_target_prob: float | None = None
    # Native FPL stat (no crosswalk needed), always a real computed number --
    # 0.0 for an outfield player, never None.
    expected_saves: float = 0.0


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


class FixtureValueBetSnapshot(BaseModel):
    """A value bet exactly as it was first surfaced before kickoff."""

    market: str
    probability: float
    implied_probability: float
    edge: float
    price: float
    bookmaker: str | None = None
    snapshotted_at: datetime
    resolved: bool
    won: bool | None = None
    final_score: str | None = None
    result_source: str | None = None


class FixtureDetail(FixtureSummary):
    score_grid: list[list[float]]
    top_scorelines: list[dict]
    corners: OverUnderPrediction
    cards: OverUnderPrediction
    # Plain expected value per team, not an O/U line -- see
    # odds/value_bets.py::predict_market_models_for_fixture's docstring for
    # why shots doesn't get the corners/cards O/U treatment.
    home_shots: float
    away_shots: float
    head_to_head: list[H2HMeeting]
    home_recent_form: list[str]
    away_recent_form: list[str]
    home_context: FixtureTeamContext
    away_context: FixtureTeamContext
    post_match: dict | None = None
    actual_stats: FixtureActualStats | None = None
    pre_match_value_bets: list[FixtureValueBetSnapshot] = []


class FixturePlayers(BaseModel):
    home_players: list[PlayerPrediction]
    away_players: list[PlayerPrediction]


class FPLManualSquadRequest(BaseModel):
    player_ids: list[int]
    bank: float = 0.0
    free_transfers: int = 1


class FPLTransferRequest(FPLManualSquadRequest):
    entry_id: int | None = None
