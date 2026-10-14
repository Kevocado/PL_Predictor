import numpy as np
import pandas as pd
import pytest

from pl_predictor.features.player_form import build_historical_start_features, current_start_features
from pl_predictor.models import player_goals
from pl_predictor.models.player_goals import predict_lineup


def test_start_features_only_use_prior_gameweeks():
    df = pd.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "element": [1] * 3,
            "kickoff_time": pd.date_range("2025-08-01", periods=3, freq="7D"),
            "minutes": [90, 15, 90],
            "starts": [1, 0, 1],
        }
    )
    rows, _ = build_historical_start_features(df)
    assert pd.isna(rows.iloc[0]["starts_last3"])
    assert rows.iloc[1]["starts_last3"] == pytest.approx(1.0)
    assert rows.iloc[2]["starts_last3"] == pytest.approx(0.5)


def test_expected_minutes_accounts_for_bench_appearances():
    prediction = predict_lineup({"starts_last5": 0.0, "sub_rate_last5": 0.5})
    assert prediction["expected_minutes"] == pytest.approx(0.5 * 39.9)


def test_current_start_features_include_unused_substitutes():
    history = pd.DataFrame({"GW": [1, 2], "minutes": [90, 0], "starts": [1, 0]})
    features = current_start_features(history)
    assert features["starts_last3"] == pytest.approx(0.5)
    assert features["sub_rate_last3"] == pytest.approx(0.0)


def test_fit_sub_rate_prior_returns_fitted_values():
    """fit_sub_rate_prior returns a prior fitted from training data."""
    prior = player_goals.fit_sub_rate_prior(seasons=["2022-23", "2023-24"])
    assert "prior_sub_rate" in prior
    assert "prior_weight" in prior
    assert 0.0 < prior["prior_sub_rate"] < 0.5
    assert prior["prior_weight"] > 0


def test_predict_lineup_uses_fitted_sub_rate_prior():
    """Players with minutes_last10>0 but sub_rate_last5==0 get nonzero sub prob."""
    prior = {"prior_sub_rate": 0.07, "prior_weight": 10.0}
    # Test with a mock lineup model that gives a known start probability < 1.0
    class MockLineupModel:
        def predict_proba(self, _):
            return np.array([[0.7, 0.3]])  # 30% start probability

    class MockCalibrator:
        def predict(self, x):
            return x  # identity

    mock_model = {"classifier": MockLineupModel(), "calibrator": MockCalibrator()}

    pred = player_goals.predict_lineup(
        {"starts_last5": 0.0, "sub_rate_last5": 0.0, "minutes_last5": 18.0, "minutes_last10": 36.0},
        lineup_model=mock_model,
        sub_rate_prior=prior,
    )
    # Should have some sub probability now (not zero)
    assert pred["expected_minutes"] > 0.0
    
    # Verify that the prior was actually used by checking expected_minutes
    # is different from the case with no minutes_last10 (where prior is not applied)
    pred_no_minutes = player_goals.predict_lineup(
        {"starts_last5": 0.0, "sub_rate_last5": 0.0, "minutes_last5": 0.0, "minutes_last10": 0.0},
        lineup_model=mock_model,
        sub_rate_prior=prior,
    )
    assert pred["expected_minutes"] > pred_no_minutes["expected_minutes"]


def test_predict_lineup_no_prior_for_players_without_recent_minutes():
    """Players with minutes_last10==0 keep zero sub prob (no prior applied)."""
    pred = player_goals.predict_lineup(
        {"starts_last5": 0.0, "sub_rate_last5": 0.0, "minutes_last5": 0.0, "minutes_last10": 0.0},
        lineup_model=None,
        sub_rate_prior={"prior_sub_rate": 0.07, "prior_weight": 10.0},
    )
    assert pred["expected_minutes"] == 0.0
