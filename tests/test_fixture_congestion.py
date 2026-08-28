"""Fixture-congestion feature tests: `rest_days.build_rest_days`'s
`other_fixtures_df` correction, and the new `fixture_congestion` module.
Uses small synthetic `matches_df`/`other_fixtures_df` frames (no network) —
`test_features.py` already covers the real-data, full-pipeline path."""

import pandas as pd

from pl_predictor.features import fixture_congestion, rest_days


def _matches(rows):
    """`rows`: list of (date, team_home, team_away). Minimal columns
    `build_rest_days`/`build_congestion_features` actually touch."""
    df = pd.DataFrame(rows, columns=["date", "team_home", "team_away"])
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = "2025-2026"
    df["goals_home"] = 1
    df["goals_away"] = 1
    return df


def test_rest_days_unaffected_when_no_other_fixtures():
    matches = _matches(
        [
            ("2025-09-01", "Arsenal", "Chelsea"),
            ("2025-09-15", "Arsenal", "Liverpool"),
        ]
    )
    result = rest_days.build_rest_days(matches)
    assert result.loc[1, "rest_days_home"] == 14


def test_rest_days_shortened_by_midweek_other_competition_fixture():
    """Arsenal plays PL on Sep 1, a Champions League match on Sep 12, then
    PL again on Sep 15 — true rest before the Sep 15 match is 3 days, not
    the 14 the PL-only calendar would show."""
    matches = _matches(
        [
            ("2025-09-01", "Arsenal", "Chelsea"),
            ("2025-09-15", "Arsenal", "Liverpool"),
        ]
    )
    other = pd.DataFrame(
        [{"team": "Arsenal", "date": pd.Timestamp("2025-09-12"), "competition": "Champions League"}]
    )
    result = rest_days.build_rest_days(matches, other_fixtures_df=other)
    assert result.loc[1, "rest_days_home"] == 3


def test_rest_days_ignores_other_fixture_for_unrelated_team():
    matches = _matches([("2025-09-01", "Arsenal", "Chelsea"), ("2025-09-15", "Arsenal", "Liverpool")])
    other = pd.DataFrame(
        [{"team": "Chelsea", "date": pd.Timestamp("2025-09-12"), "competition": "Champions League"}]
    )
    result = rest_days.build_rest_days(matches, other_fixtures_df=other)
    assert result.loc[1, "rest_days_home"] == 14


def test_games_last_14_days_counts_across_competitions():
    matches = _matches(
        [
            ("2025-09-01", "Arsenal", "Chelsea"),
            ("2025-09-15", "Arsenal", "Liverpool"),
        ]
    )
    other = pd.DataFrame(
        [
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-04"), "competition": "EFL Cup"},
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-12"), "competition": "Champions League"},
        ]
    )
    result = fixture_congestion.build_congestion_features(matches, other_fixtures_df=other)
    # Sep 1 PL match + EFL Cup (Sep 4) + CL (Sep 12) all fall in the 14 days
    # before Sep 15 -> 3 prior games.
    assert result.loc[1, "games_last_14_days_home"] == 3
    # Nothing precedes the Sep 1 fixture.
    assert result.loc[0, "games_last_14_days_home"] == 0


def test_european_fixture_last_4_days_flag():
    matches = _matches([("2025-09-01", "Arsenal", "Chelsea"), ("2025-09-15", "Arsenal", "Liverpool")])
    other = pd.DataFrame(
        [
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-04"), "competition": "EFL Cup"},
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-12"), "competition": "Champions League"},
        ]
    )
    result = fixture_congestion.build_congestion_features(matches, other_fixtures_df=other)
    # EFL Cup isn't a European competition -> no flag for the Sep 1 fixture's
    # (nonexistent) follow-up; the Sep 15 fixture is within 4 days of the
    # Sep 12 Champions League match -> flagged.
    assert result.loc[1, "european_fixture_last_4_days_home"] == 1
    assert result.loc[0, "european_fixture_last_4_days_home"] == 0


def test_congestion_features_never_use_same_or_future_row():
    """No-lookahead guarantee: a fixture on the *same* day as another
    competition's match (or a future one) must not count itself."""
    matches = _matches([("2025-09-15", "Arsenal", "Liverpool")])
    other = pd.DataFrame(
        [
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-15"), "competition": "Champions League"},
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-20"), "competition": "Champions League"},
        ]
    )
    result = fixture_congestion.build_congestion_features(matches, other_fixtures_df=other)
    assert result.loc[0, "games_last_14_days_home"] == 0
    assert result.loc[0, "european_fixture_last_4_days_home"] == 0
