"""rest_days.py — fixture-congestion features: days since each side's
previous match, and whether this is a team's first match of the season."""

from __future__ import annotations

import pandas as pd

from .rolling_form import to_team_perspective


def _team_match_calendar(matches_df: pd.DataFrame, other_fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (team, date) a team played *any* match — Premier League
    plus, when supplied, Champions League/Europa League/Conference
    League/FA Cup/EFL Cup dates from
    `data.other_competitions.get_team_fixture_calendar`. This is what makes
    `rest_days` reflect a team's true previous match rather than just its
    previous PL one."""
    calendar = to_team_perspective(matches_df)[["team", "date"]].copy()
    if other_fixtures_df is not None and not other_fixtures_df.empty:
        extra = other_fixtures_df[["team", "date"]].copy()
        extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
        calendar = pd.concat([calendar, extra], ignore_index=True)
    return calendar.drop_duplicates().sort_values(["team", "date"]).reset_index(drop=True)


def build_rest_days(matches_df: pd.DataFrame, other_fixtures_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns `rest_days_home`, `rest_days_away`,
    `is_first_match_of_season_home`, `is_first_match_of_season_away`,
    aligned to `matches_df`'s index. `other_fixtures_df` is optional
    (omitting it, or passing an empty frame, reproduces the exact PL-only
    behavior this function always had) — see `_team_match_calendar`."""
    long_df = to_team_perspective(matches_df).sort_values(["team", "date"])

    calendar = _team_match_calendar(matches_df, other_fixtures_df)
    calendar["prev_date"] = calendar.groupby("team", sort=False)["date"].shift(1)
    prev_date_by_team_date = calendar.set_index(["team", "date"])["prev_date"]

    long_df["prev_date"] = prev_date_by_team_date.reindex(
        pd.MultiIndex.from_arrays([long_df["team"], long_df["date"]])
    ).to_numpy()
    long_df["rest_days"] = (long_df["date"] - long_df["prev_date"]).dt.days

    grouped = long_df.groupby("team", sort=False)
    long_df["is_first_match_of_season"] = grouped["season"].transform(
        lambda s: s != s.shift(1)
    )

    keyed = long_df.set_index(["date", "team", "was_home"])

    def _lookup(dates, teams, was_home_flag):
        idx = list(zip(dates, teams, [was_home_flag] * len(dates)))
        return keyed.reindex(idx)

    home_lookup = _lookup(matches_df["date"], matches_df["team_home"], True)
    away_lookup = _lookup(matches_df["date"], matches_df["team_away"], False)

    return pd.DataFrame(
        {
            "rest_days_home": home_lookup["rest_days"].to_numpy(),
            "rest_days_away": away_lookup["rest_days"].to_numpy(),
            "is_first_match_of_season_home": home_lookup["is_first_match_of_season"].to_numpy(),
            "is_first_match_of_season_away": away_lookup["is_first_match_of_season"].to_numpy(),
        },
        index=matches_df.index,
    )
