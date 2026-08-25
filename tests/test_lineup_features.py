import pandas as pd
import pytest

from pl_predictor.features.player_form import build_historical_start_features, current_start_features
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
