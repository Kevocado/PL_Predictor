import pandas as pd
import pytest

from pl_predictor.odds import value_bets


def _odds_frame(home_price: float = 2.25) -> pd.DataFrame:
    rows = [
        ("h2h", "Home", home_price, None, "best-home-book"),
        ("h2h", "Draw", 4.0, None, "draw-book"),
        ("h2h", "Away", 5.0, None, "away-book"),
        ("totals", "Over", 2.0, 2.5, "over-book"),
        ("totals", "Under", 2.0, 2.5, "under-book"),
    ]
    return pd.DataFrame(rows, columns=["market", "outcome_name", "price", "point", "bookmaker"]).assign(
        event_id="event-1", odds_fetched_at=pd.Timestamp.now(tz="UTC")
    )


def _fixtures_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "commence_time": pd.Timestamp("2026-08-30T15:00:00Z"),
                "team_home": "Home",
                "team_away": "Away",
            }
        ]
    )


def test_recommendation_uses_one_best_qualified_quote(monkeypatch):
    monkeypatch.setattr(value_bets, "_devig_h2h", lambda *_: {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})
    monkeypatch.setattr(
        value_bets.scoreline,
        "predict_fixtures_batch",
        lambda *_, **__: [
            {
                "home_win": 0.90,
                "draw": 0.16,
                "away_win": 0.14,
                "btts_yes": 0.5,
                "over_2_5": 0.5,
                "under_2_5": 0.5,
                "top_scorelines": [{"home": 2, "away": 0}],
                "fallback": False,
                "data_confidence": "established",
            }
        ],
    )
    row = value_bets.build_value_bet_table(_fixtures_frame(), _odds_frame(), models={"scoreline": object()}).iloc[0]
    assert row["recommended_market"] == "home_win"
    assert row["recommended_price"] == pytest.approx(2.25)
    assert row["recommended_bookmaker"] == "best-home-book"
    assert "parlay" not in row.index


def test_recommendation_excludes_tail_prices_and_fallback(monkeypatch):
    monkeypatch.setattr(value_bets, "_devig_h2h", lambda *_: {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})
    monkeypatch.setattr(
        value_bets.scoreline,
        "predict_fixtures_batch",
        lambda *_, **__: [
            {
                "home_win": 0.70,
                "draw": 0.16,
                "away_win": 0.14,
                "btts_yes": 0.5,
                "over_2_5": 0.2,
                "under_2_5": 0.8,
                "top_scorelines": [{"home": 2, "away": 0}],
                "fallback": True,
                "data_confidence": "new",
            }
        ],
    )
    row = value_bets.build_value_bet_table(_fixtures_frame(), _odds_frame(home_price=7.0), models={"scoreline": object()}).iloc[0]
    assert row["recommended_market"] is None


def test_stale_odds_do_not_create_a_value_recommendation(monkeypatch):
    monkeypatch.setattr(value_bets, "_devig_h2h", lambda *_: {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})
    monkeypatch.setattr(
        value_bets.scoreline,
        "predict_fixtures_batch",
        lambda *_, **__: [{"home_win": 0.90, "draw": 0.06, "away_win": 0.04, "btts_yes": 0.5, "over_2_5": 0.5, "under_2_5": 0.5, "top_scorelines": [{"home": 2, "away": 0}], "fallback": False, "data_confidence": "established"}],
    )
    odds = _odds_frame().assign(odds_fetched_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))

    row = value_bets.build_value_bet_table(_fixtures_frame(), odds, models={"scoreline": object()}).iloc[0]

    assert bool(row["odds_is_stale"]) is True
    assert row["value_bet_flags"] == []
    assert row["recommended_market"] is None
