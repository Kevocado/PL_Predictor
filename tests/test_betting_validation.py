import pytest

from pl_predictor.evaluate import betting_validation


def test_yield_summary_uses_flat_stake_returns():
    selections = [
        {"price": 2.0, "won": True},
        {"price": 3.0, "won": False},
        {"price": 1.5, "won": True},
    ]

    summary = betting_validation._yield_summary(selections)

    assert summary == {
        "bets": 3,
        "wins": 2,
        "win_rate": pytest.approx(66.66666666666666),
        "yield": pytest.approx(16.666666666666664),
    }


def test_yield_interval_requires_at_least_two_bets():
    assert betting_validation._yield_interval([]) is None
    assert betting_validation._yield_interval([{"price": 2.0, "won": True}]) is None


def test_market_and_odds_breakdowns_label_live_markets():
    assert betting_validation._market_label({"selection": "home_win"}) == "Match result"
    assert betting_validation._market_label({"selection": "over_2_5"}) == "Goals O/U 2.5"
    assert betting_validation._odds_band({"price": 1.9}) == "Under +100"
    assert betting_validation._odds_band({"price": 2.5}) == "+100 to +199"
    assert betting_validation._odds_band({"price": 3.0}) == "+200 or longer"
