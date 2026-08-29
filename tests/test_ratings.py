"""team_rating_timeseries's gameweek join — matches_df (football-data.co.uk,
which this function replays) has no gameweek column, only fd_org_matches
(football-data.org) does, so a real match's rating point gets its gameweek
by matching team pair + nearest date across the two sources rather than
assuming identical dates between providers."""

import pandas as pd

from pl_predictor.features import ratings


def _matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-16"), "team_home": "Arsenal", "team_away": "Wolves", "ftr": "H", "goals_home": 2, "goals_away": 0},
            {"date": pd.Timestamp("2026-08-23"), "team_home": "Wolves", "team_away": "Arsenal", "ftr": "A", "goals_home": 0, "goals_away": 1},
        ]
    )


def _fd_org_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_home": "Arsenal", "team_away": "Wolves", "commence_time": pd.Timestamp("2026-08-16"), "matchday": 1},
            # One day off from matches_df's own date for the same fixture — still within the join's tolerance.
            {"team_home": "Wolves", "team_away": "Arsenal", "commence_time": pd.Timestamp("2026-08-24"), "matchday": 2},
        ]
    )


def test_attaches_gameweek_by_nearest_date_match():
    history = ratings.team_rating_timeseries(_matches_df(), fd_org_matches=_fd_org_matches())

    gw1 = history[history["date"] == pd.Timestamp("2026-08-16")]["gameweek"].dropna().unique()
    assert list(gw1) == [1]
    gw2 = history[history["date"] == pd.Timestamp("2026-08-23")]["gameweek"].dropna().unique()
    assert list(gw2) == [2]


def test_gameweek_is_none_without_fd_org_matches():
    history = ratings.team_rating_timeseries(_matches_df())
    assert history["gameweek"].isna().all()


def test_gameweek_is_none_beyond_tolerance_window():
    fd_org_matches = _fd_org_matches().copy()
    fd_org_matches.loc[0, "commence_time"] = pd.Timestamp("2026-08-25")  # 9 days from the real match date — too far

    history = ratings.team_rating_timeseries(_matches_df(), fd_org_matches=fd_org_matches)

    gw1_rows = history[history["date"] == pd.Timestamp("2026-08-16")]
    assert gw1_rows["gameweek"].isna().all()
