import pandas as pd
import pytest

from pl_predictor.evaluate import backtest


def test_historical_replay_records_qualified_de_vigged_selection(monkeypatch):
    fixtures = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-08-01"),
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
                "b365_h": 2.0,
                "b365_d": 3.5,
                "b365_a": 4.0,
                "b365>2.5": 1.9,
                "b365<2.5": 2.0,
            }
        ]
    )
    monkeypatch.setattr(
        backtest,
        "_precompute_predictions",
        lambda _model, frame, market_overrides=None: {
            frame.index[0]: {
                "home_win": 0.40,
                "draw": 0.30,
                "away_win": 0.30,
                "over_2_5": 0.75,
                "under_2_5": 0.25,
                "fallback": False,
            }
        },
    )
    selections = []

    replay = backtest.build_value_bet_backtest(
        fixtures,
        model=object(),
        start_date="2025-08-01",
        end_date="2025-08-01",
        staking="flat",
        selections=selections,
    )

    assert replay.results()["Total Bets"] == 1
    assert selections == [
        {
            "date": "2025-08-01",
            "fixture": "Arsenal 2-1 Chelsea",
            "selection": "over_2_5",
            "price": 1.9,
            "model_probability": 0.75,
            "implied_probability": selections[0]["implied_probability"],
            "edge": selections[0]["edge"],
            "won": True,
        }
    ]
    assert selections[0]["edge"] > 0.05


def test_bootstrap_drawdown_distribution_empty_selections_returns_none_stats():
    result = backtest.bootstrap_drawdown_distribution([])
    assert result["n_trials"] == 0
    assert result["max_drawdown_pct"]["mean"] is None
    assert result["roi_pct"]["mean"] is None


def test_bootstrap_drawdown_distribution_is_deterministic_for_a_fixed_seed():
    selections = [
        {"price": 2.0, "model_probability": 0.6},
        {"price": 3.0, "model_probability": 0.4},
        {"price": 1.8, "model_probability": 0.7},
    ]
    first = backtest.bootstrap_drawdown_distribution(selections, staking="flat", flat_stake=5.0, n_trials=200, seed=42)
    second = backtest.bootstrap_drawdown_distribution(selections, staking="flat", flat_stake=5.0, n_trials=200, seed=42)
    assert first == second
    assert first["n_trials"] == 200
    assert 0.0 <= first["max_drawdown_pct"]["mean"] <= 100.0
    # p95 drawdown is a worse (or equal) case than the median across the same trials.
    assert first["max_drawdown_pct"]["p95"] >= first["max_drawdown_pct"]["median"]


def test_bootstrap_drawdown_distribution_a_sure_loser_always_hits_full_stake_drawdown():
    # model_probability=0 means every simulated draw loses -- deterministic
    # worst case, useful as a sanity bound on the machinery itself.
    selections = [{"price": 2.0, "model_probability": 0.0}] * 5
    result = backtest.bootstrap_drawdown_distribution(selections, staking="flat", flat_stake=10.0, n_trials=50, seed=1)
    assert result["max_drawdown_pct"]["worst"] == pytest.approx(50.0)
    assert result["roi_pct"]["mean"] == pytest.approx(-50.0)


def test_low_confidence_prediction_needs_a_bigger_edge_to_be_selected(monkeypatch):
    fixtures = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-08-01"),
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
                "b365_h": 2.0,
                "b365_d": 3.5,
                "b365_a": 4.0,
                "b365>2.5": 1.9,
                "b365<2.5": 2.0,
            }
        ]
    )
    # Same 0.08-ish edge shape as test_historical_replay_records_qualified_
    # de_vigged_selection above, but tagged "new" -- must NOT be selected,
    # since 0.08 clears the flat 5% default but not "new"'s 2.5x (12.5%).
    monkeypatch.setattr(
        backtest,
        "_precompute_predictions",
        lambda _model, frame, market_overrides=None: {
            frame.index[0]: {
                "home_win": 0.48,
                "draw": 0.30,
                "away_win": 0.22,
                "over_2_5": 0.59,
                "under_2_5": 0.41,
                "fallback": False,
                "data_confidence": "new",
            }
        },
    )
    selections = []

    replay = backtest.build_value_bet_backtest(
        fixtures,
        model=object(),
        start_date="2025-08-01",
        end_date="2025-08-01",
        staking="flat",
        selections=selections,
    )

    assert replay.results()["Total Bets"] == 0
    assert selections == []
