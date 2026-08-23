"""table_context.py — league-table stakes: points off the top-4 cutoff,
points off the relegation zone, games played so far this season.

Rolling-form features (rolling_form.py) capture how a team's been
*playing*; this captures where they actually *stand* — a mid-table team
with nothing left to play for behaves differently from one fighting
relegation, even with identical recent form. Unlike every other feature in
this package, this is cross-sectional rather than per-team: a team's
`points_off_top4` as of a given date needs *all 20 teams'* cumulative
points as of that date, not just the row's own two teams. Uses the same
`merge_asof` idiom `xg_form.py::attach_xg_features` uses to join an
independently-scheduled data source onto `matches_df` by date, just applied
across the full season roster instead of one row's two teams.

Sign convention: `points_off_top4` is positive when a team is below the
top-4 cutoff by that many points (0 or negative if they're in the top 4
already); `points_off_relegation` is positive when a team is clear of the
drop zone by that many points (negative if they're actually in it).
"""

from __future__ import annotations

import pandas as pd

from . import rolling_form

FEATURE_COLS = [
    "home_points_off_top4",
    "home_points_off_relegation",
    "home_games_played_this_season",
    "away_points_off_top4",
    "away_points_off_relegation",
    "away_games_played_this_season",
]

_TOP4_RANK = 4
_RELEGATION_SAFE_RANK = 17  # last non-relegation place in a 20-team league


def _stakes_from_ranked(g: pd.DataFrame, points_col: str) -> pd.DataFrame:
    """`g` must already be sorted best-team-first (descending points, then
    descending goal difference — the same order
    `models/projected_table.py::compute_standings` already produces).
    Shared by both the historical (per-checkpoint) and live-serving paths so
    the top4/relegation arithmetic is defined exactly once.

    Uses `.iloc` (positional) rather than resetting `g`'s index: this is
    called once per (season, date) checkpoint via `groupby(...).apply(...)`
    in `attach_table_context_features`, and resetting the index there would
    collapse every group's index down to the same `0..n-1` range — fine
    within one group, but a landmine once pandas concatenates all the
    groups back together (thousands of duplicate index values, since every
    checkpoint's row 0 becomes index 0)."""
    n = len(g)
    top4_points = g[points_col].iloc[_TOP4_RANK - 1] if n >= _TOP4_RANK else g[points_col].min()
    cutoff_idx = min(_RELEGATION_SAFE_RANK - 1, n - 1)
    releg_points = g[points_col].iloc[cutoff_idx] if n > 0 else 0
    g = g.copy()
    g["points_off_top4"] = top4_points - g[points_col]
    g["points_off_relegation"] = g[points_col] - releg_points
    return g


def _cumulative_after_each(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Each team's cumulative (points, played, goal_diff) *after* each of
    its own matches, within its own season — one row per team-match."""
    long_df = rolling_form.to_team_perspective(matches_df)
    long_df = long_df.sort_values(["season", "team", "date"]).reset_index(drop=True)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    grouped = long_df.groupby(["season", "team"], sort=False)
    long_df["cum_points"] = grouped["points"].cumsum()
    long_df["cum_played"] = grouped.cumcount() + 1
    long_df["cum_goal_diff"] = grouped["goal_diff"].cumsum()
    return long_df[["season", "team", "date", "cum_points", "cum_played", "cum_goal_diff"]]


def _season_team_roster(matches_df: pd.DataFrame) -> pd.DataFrame:
    home = matches_df[["season", "team_home"]].rename(columns={"team_home": "team"})
    away = matches_df[["season", "team_away"]].rename(columns={"team_away": "team"})
    return pd.concat([home, away], ignore_index=True).drop_duplicates().reset_index(drop=True)


def attach_table_context_features(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Shift(1)-safe: every row's stakes reflect the table strictly before
    that match kicked off. Same contract as `xg_form.attach_xg_features` —
    a frame aligned to `matches_df`'s row order."""
    cum = _cumulative_after_each(matches_df).sort_values("date")
    roster = _season_team_roster(matches_df)
    checkpoints = matches_df[["season", "date"]].drop_duplicates()

    snapshot = checkpoints.merge(roster, on="season").sort_values("date")
    snapshot = pd.merge_asof(
        snapshot,
        cum[["season", "team", "date", "cum_points", "cum_played", "cum_goal_diff"]],
        on="date",
        by=["season", "team"],
        direction="backward",
        allow_exact_matches=False,
    )
    for col in ("cum_points", "cum_played", "cum_goal_diff"):
        snapshot[col] = snapshot[col].fillna(0)

    def _rank_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["cum_points", "cum_goal_diff"], ascending=False)
        return _stakes_from_ranked(g, points_col="cum_points")

    # pandas' groupby(...).apply(group_keys=False) drops the grouping
    # columns from each sub-frame before calling the function (confirmed on
    # pandas 3.0), so `season`/`date` need to be restored afterward from the
    # original index — `_rank_group` only reorders rows, never touches the
    # index, so `snapshot.loc[ranked.index]` lines back up correctly.
    ranked = snapshot.groupby(["season", "date"], group_keys=False).apply(_rank_group)
    ranked[["season", "date"]] = snapshot.loc[ranked.index, ["season", "date"]]
    stakes = ranked.rename(columns={"cum_played": "games_played_this_season"})[
        ["season", "date", "team", "points_off_top4", "points_off_relegation", "games_played_this_season"]
    ]

    left = matches_df[["date", "season", "team_home", "team_away"]].reset_index()
    home = stakes.rename(
        columns={
            "team": "team_home",
            "points_off_top4": "home_points_off_top4",
            "points_off_relegation": "home_points_off_relegation",
            "games_played_this_season": "home_games_played_this_season",
        }
    )
    away = stakes.rename(
        columns={
            "team": "team_away",
            "points_off_top4": "away_points_off_top4",
            "points_off_relegation": "away_points_off_relegation",
            "games_played_this_season": "away_games_played_this_season",
        }
    )
    merged = left.merge(home, on=["season", "date", "team_home"], how="left").merge(
        away, on=["season", "date", "team_away"], how="left"
    )
    merged = merged.set_index("index").reindex(matches_df.index)
    out = merged[FEATURE_COLS].fillna(0)
    return out, FEATURE_COLS


def live_stakes(standings: pd.DataFrame) -> pd.DataFrame:
    """Live-serving equivalent: takes `models/projected_table.py::
    compute_standings`'s output directly (already sorted best-first by
    points/goal_diff) and adds `points_off_top4`/`points_off_relegation` —
    indexed by team, ready for `FixtureFeatureContext.build_row` to look up.
    No shift(1) needed since "now" already reflects every played match."""
    if standings.empty:
        return standings
    ranked = _stakes_from_ranked(standings, points_col="points")
    return ranked.set_index("team")
