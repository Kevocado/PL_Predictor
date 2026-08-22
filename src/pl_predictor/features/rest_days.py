"""rest_days.py — fixture-congestion features: days since each side's
previous match, and whether this is a team's first match of the season."""

from __future__ import annotations

import pandas as pd

from .rolling_form import to_team_perspective


def build_rest_days(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Returns `rest_days_home`, `rest_days_away`,
    `is_first_match_of_season_home`, `is_first_match_of_season_away`,
    aligned to `matches_df`'s index."""
    long_df = to_team_perspective(matches_df).sort_values(["team", "date"])

    grouped = long_df.groupby("team", sort=False)
    long_df["prev_date"] = grouped["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_date"]).dt.days
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
