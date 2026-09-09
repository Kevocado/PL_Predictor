"""Unit tests for FixtureSummary's computed predicted_result/draw_signal
fields (see outcomes.py for the underlying agreement rule)."""

from datetime import datetime, timezone

from pl_predictor.api.schemas import FixtureSummary, MarketEdge


def _summary(top_scoreline: str, home_win: float, draw: float, away_win: float) -> FixtureSummary:
    return FixtureSummary(
        event_id="e1",
        commence_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        team_home="Arsenal",
        team_away="Chelsea",
        home_win=MarketEdge(prob=home_win),
        draw=MarketEdge(prob=draw),
        away_win=MarketEdge(prob=away_win),
        over_2_5=MarketEdge(prob=0.5),
        under_2_5=MarketEdge(prob=0.5),
        btts_yes_prob=0.5,
        top_scoreline=top_scoreline,
        is_fallback_prediction=False,
        value_bet_flags=[],
        has_live_odds=False,
    )


def test_predicted_result_and_draw_signal_default_to_plain_argmax():
    summary = _summary("2-1", 0.50, 0.25, 0.25)
    assert summary.predicted_result == "home_win"
    assert summary.draw_signal is False


def test_draw_signal_true_when_scoreline_and_percentage_models_agree():
    summary = _summary("1-1", 0.40, 0.30, 0.30)
    assert summary.predicted_result == "draw"
    assert summary.draw_signal is True


def test_draw_signal_false_when_draw_probability_below_agreement_threshold():
    summary = _summary("1-1", 0.40, 0.15, 0.45)
    assert summary.predicted_result == "away_win"
    assert summary.draw_signal is False
