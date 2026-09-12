import pandas as pd

from pl_predictor.features.build import TARGET_COLS, _add_targets


def test_add_targets_carries_home_and_away_shots_through_unchanged():
    matches_df = pd.DataFrame([{"hs": 14, "as": 9}, {"hs": 7, "as": 18}])

    df = _add_targets(matches_df)

    assert df["home_shots"].tolist() == [14, 7]
    assert df["away_shots"].tolist() == [9, 18]


def test_add_targets_defaults_missing_shots_columns_to_zero():
    matches_df = pd.DataFrame([{"hc": 5, "ac": 3}])

    df = _add_targets(matches_df)

    assert df["home_shots"].tolist() == [0]
    assert df["away_shots"].tolist() == [0]


def test_shots_targets_are_excluded_from_feature_columns():
    assert "home_shots" in TARGET_COLS
    assert "away_shots" in TARGET_COLS
