import pandas as pd
import pytest

from pl_predictor.features import xg_form


def _understat_df():
    dates = pd.date_range("2023-08-01", periods=3, freq="7D")
    return pd.DataFrame(
        [
            {"date": dates[0], "team_home": "A", "team_away": "B", "xg_home": 1.0, "xg_away": 0.5, "goals_home": 1, "goals_away": 0},
            {"date": dates[1], "team_home": "B", "team_away": "A", "xg_home": 0.7, "xg_away": 2.0, "goals_home": 0, "goals_away": 2},
            {"date": dates[2], "team_home": "A", "team_away": "B", "xg_home": 3.0, "xg_away": 0.3, "goals_home": 3, "goals_away": 0},
        ]
    )


def test_attach_xg_features_includes_the_most_recent_prior_match():
    """Regression test for a confirmed real-data bug: a double no-lookahead
    exclusion (shift(1) on the source frame *and* merge_asof's own
    allow_exact_matches=False) made every joined value skip the source
    team's most recent actual match, landing one full match stale. The
    no-lookahead guarantee belongs entirely to merge_asof's backward/
    exclude-exact-match join here; build_rolling_xg must not shift on top
    of it."""
    understat_df = _understat_df()
    # A fixture strictly after all three known matches — its rolling
    # feature should reflect ALL of team A's known xG_for values (1.0, 2.0,
    # 3.0), not stop one match short.
    target_date = understat_df["date"].iloc[-1] + pd.Timedelta(days=7)
    matches_df = pd.DataFrame([{"date": target_date, "team_home": "A", "team_away": "B"}])

    out, _ = xg_form.attach_xg_features(matches_df, understat_df)

    assert out.loc[0, "home_xg_for_last_5"] == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def test_attach_xg_features_excludes_same_date_match():
    understat_df = _understat_df()
    same_date_fixture = pd.DataFrame([{"date": understat_df["date"].iloc[2], "team_home": "A", "team_away": "B"}])

    out, _ = xg_form.attach_xg_features(same_date_fixture, understat_df)

    # Must not see the same-date match's own xG (3.0) — only the two before it.
    assert out.loc[0, "home_xg_for_last_5"] == pytest.approx((1.0 + 2.0) / 2)
