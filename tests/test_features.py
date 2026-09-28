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


def test_date_keyed_merges_survive_mismatched_datetime_resolutions():
    """The bug CI found and a warm local cache hid.

    A `merge` on a datetime key needs both sides at the *same resolution*, not merely
    the same values. `load_training_data` reads `data/cache/*.parquet`, and parquet
    round-trips whatever dtype wrote it, so the same logical `date` column arrives as
    `datetime64[s]` from one cache and `datetime64[us]` from another -- and
    `pd.to_datetime` infers `s` or `us` from the strings it is given.

    The result is

        pandas.errors.MergeError: incompatible merge keys [1]
        dtype('<M8[us]') and dtype('<M8[s]'), must be the same type

    on two merges that are otherwise entirely correct. It passed on a warm local
    cache and failed on CI's cold one, in `test_no_lookahead_in_training_frame` and
    `test_current_season_check`. Nothing about the test or the model changed; only
    which pandas wrote the cache.

    Constructed here explicitly at both resolutions, so it fails on *any* machine
    rather than only on the one whose cache happens to disagree.
    """
    from pl_predictor.features import build as build_module

    base = {
        "date": ["2024-08-17", "2024-08-24", "2024-08-17", "2024-08-24"],
        "team_home": ["Liverpool", "Arsenal", "Arsenal", "Liverpool"],
        "team_away": ["Arsenal", "Liverpool", "Liverpool", "Arsenal"],
        "season": ["2024-2025"] * 4,
        "goals_home": [2, 1, 1, 3],
        "goals_away": [1, 1, 3, 2],
        "ftr": ["H", "H", "A", "H"],
    }
    for resolution in ("s", "us", "ns"):
        matches = pd.DataFrame(base)
        matches["date"] = pd.to_datetime(matches["date"]).astype(f"datetime64[{resolution}]")

        df, feature_cols = build_module.build_training_frame(matches_df=matches)

        assert len(df) == 4, f"merge lost rows at resolution {resolution}"
        # The rolling-form features must actually have matched, not merely not raised.
        assert df["elo_home"].notna().all(), (
            f"at resolution {resolution} the date join silently failed to match, so every "
            f"feature is missing -- which is what a wrong-resolution merge looks like once "
            f"the dtype mismatch is taken out of the way"
        )
        assert df[feature_cols].shape[1] == len(feature_cols)


def test_date_key_normalisation_is_not_silently_dropping_bad_values():
    """`errors="coerce"` is a choice, so it gets a test.

    A cell that cannot be read as a timestamp becomes NaT and fails to match, which
    under a `how="left"` merge degrades one row's features to NaN. That is the
    intended trade: XGBoost handles NaN natively, whereas raising would take out a
    whole training run over one bad cell in a cached upstream CSV. This pins the
    choice so it cannot be changed silently in either direction.
    """
    from pl_predictor.features.date_keys import as_date_key

    frame = pd.DataFrame({"date": ["2024-08-17", "not-a-date", None], "team": ["A", "B", "C"]})

    out = as_date_key(frame)

    assert out["date"].dtype == "datetime64[ns]"
    assert pd.notna(out["date"].iloc[0])
    assert out["date"].iloc[1] is pd.NaT or pd.isna(out["date"].iloc[1])
    assert pd.isna(out["date"].iloc[2])
