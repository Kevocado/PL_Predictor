"""streaks.py — current win/loss streak length.

`features/rolling_form.py`'s `last_3/5/10_*` windows are plain averages, so
a team on W-L-W-L-W looks identical to one on a genuine 5-game win streak —
same average, very different momentum. This is a single signed integer per
team: +N for an N-game win streak, -N for an N-game losing streak, 0 if the
most recent result was a draw or there's no history yet. A draw resets the
streak rather than getting its own streak type — momentum here is really
"on a run" vs. "not," not a three-way state.
"""

from __future__ import annotations

import pandas as pd

from . import rolling_form

FEATURE_COLS = ["home_current_streak", "away_current_streak"]


def _streak_after_each(points: list[int]) -> list[int]:
    """`points` in date order (3=W, 1=D, 0=L). Returns the signed streak
    *after* each match (inclusive of that match's own result) — callers
    needing a shift(1)-safe "before this match" value should shift the
    result by one themselves (see `attach_streak_features`)."""
    streaks = []
    current = 0
    for p in points:
        if p == 3:
            current = current + 1 if current > 0 else 1
        elif p == 0:
            current = current - 1 if current < 0 else -1
        else:
            current = 0
        streaks.append(current)
    return streaks


def attach_streak_features(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Shift(1)-safe per-row streak feature, same contract as
    `xg_form.attach_xg_features`: a frame aligned to `matches_df`'s row
    order with `home_current_streak`/`away_current_streak` columns. Unlike
    xg_form's `merge_asof` (an independent, differently-scheduled data
    source), this joins on the exact `(date, team)` key since the streak is
    derived from `matches_df` itself via `rolling_form.to_team_perspective`."""
    long_df = rolling_form.to_team_perspective(matches_df)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    long_df["current_streak"] = long_df.groupby("team", sort=False)["points"].transform(
        lambda s: pd.Series(_streak_after_each(s.tolist()), index=s.index).shift(1).fillna(0)
    )

    home_join = (
        long_df[long_df["was_home"]]
        .set_index(["date", "team"])["current_streak"]
        .rename("home_current_streak")
        .reset_index()
        .rename(columns={"team": "team_home"})
    )
    away_join = (
        long_df[~long_df["was_home"]]
        .set_index(["date", "team"])["current_streak"]
        .rename("away_current_streak")
        .reset_index()
        .rename(columns={"team": "team_away"})
    )

    left = matches_df[["date", "team_home", "team_away"]].reset_index()
    merged = left.merge(home_join, on=["date", "team_home"], how="left").merge(
        away_join, on=["date", "team_away"], how="left"
    )
    merged = merged.set_index("index").reindex(matches_df.index)
    out = merged[FEATURE_COLS].fillna(0)
    return out, FEATURE_COLS


def latest_streaks(matches_df: pd.DataFrame) -> pd.Series:
    """Every team's streak *right now*, i.e. after its most recent played
    match — no shift(1) needed since "now" is genuinely after every known
    result. Indexed by team. Computed once for the whole league in a single
    pass (same idiom as `xg_form.latest_xg_form`) rather than recomputing
    per fixture — see `FixtureFeatureContext.__init__`, which calls this
    once and looks teams up from the result in `build_row`."""
    long_df = rolling_form.to_team_perspective(matches_df)
    long_df = long_df.sort_values(["team", "date"])
    return long_df.groupby("team", sort=False)["points"].apply(lambda s: _streak_after_each(s.tolist())[-1])
