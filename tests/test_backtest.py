import pandas as pd

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
