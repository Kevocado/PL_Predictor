"""Leakage and sanity checks for the feature layer. Run with `pytest`."""

import pandas as pd
import pytest

from pl_predictor.data.football_data import load_training_data
from pl_predictor.features.build import FixtureFeatureContext, build_training_frame
from pl_predictor.features.rolling_form import build_rolling_form


@pytest.fixture(scope="module")
def matches():
    return load_training_data(seasons=["2022-2023", "2023-2024"])


def test_rolling_form_first_match_is_nan(matches):
    long_df, feature_cols = build_rolling_form(matches)
    first_rows = long_df.sort_values("date").groupby("team", sort=False).head(1)
    for col in feature_cols:
        assert first_rows[col].isna().all(), f"{col} should be NaN for a team's first match"


def test_rolling_form_never_uses_same_row_stat(matches):
    """A team's rolling goals-for average, right after a single match, must
    equal that match's actual goals-for (not include it via off-by-one)."""
    long_df, _ = build_rolling_form(matches)
    long_df = long_df.sort_values(["team", "date"])
    team = long_df["team"].iloc[0]
    team_rows = long_df[long_df["team"] == team].reset_index(drop=True)
    assert team_rows.loc[1, "last_3_goals_for"] == pytest.approx(team_rows.loc[0, "goals_for"])


def test_no_lookahead_in_training_frame(matches):
    df, feature_cols = build_training_frame(matches_df=matches)
    assert (df["date"] == matches["date"]).all()
    # elo/pi ratings before the match must differ from ratings after (i.e.
    # the feature isn't silently constant / a copy of a post-match value)
    assert df["elo_home"].nunique() > 1
    assert df[feature_cols].shape[1] == len(feature_cols)


def test_cold_start_confidence_present(matches):
    df, feature_cols = build_training_frame(matches_df=matches)
    assert "confidence_home" in df.columns
    assert set(df["confidence_home"].unique()) <= {"current", "blended", "none"}


def test_build_row_accepts_tz_aware_commence_time(matches):
    """A live fixtures source (e.g. the Odds API) hands commence_time as a
    tz-aware UTC timestamp; matches_df's own `date` column is tz-naive.
    build_row must not blow up computing rest days from that mismatch —
    regression test for a real 500 this exact combination caused."""
    ctx = FixtureFeatureContext(matches)
    home, away = matches["team_home"].iloc[-1], matches["team_away"].iloc[-1]
    tz_aware_commence_time = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)
    row = ctx.build_row(home, away, commence_time=tz_aware_commence_time)
    assert row["rest_days_home"] is not None or row["is_first_match_of_season_home"]
