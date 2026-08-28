"""fixture_congestion.py — squad-rotation/fatigue signals derived from a
team's full match calendar (Premier League plus Champions League/Europa
League/Conference League/FA Cup/EFL Cup, from
`data.other_competitions.get_team_fixture_calendar`), distinct from
`rest_days.py`'s single "days since last match" number:

- `games_last_14_days_{home,away}`: how many matches (any competition) a
  team played in the two weeks before this fixture — a congestion/rotation
  proxy a single rest-days number can't capture (a team that's played 4
  games in 14 days is more fatigued than one on the same rest-days count
  coming off one postponed-then-rescheduled gap).
- `european_fixture_last_4_days_{home,away}`: 1/0, whether a team played a
  Champions/Europa/Conference League match in the 4 days before this
  fixture — the classic Thursday-Europe-to-Sunday-PL turnaround.

Both are computed here but NOT unconditionally added to `features/build.py`'s
`feature_cols` — see that module's own precedent for `stakes_cols`/
`situation_cols`: a new feature only ships once a walk-forward check shows
it actually helps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rolling_form import to_team_perspective

GAMES_WINDOW_DAYS = 14
EUROPEAN_WINDOW_DAYS = 4
EUROPEAN_COMPETITIONS = {"Champions League", "Europa League", "Conference League"}


def _full_calendar(matches_df: pd.DataFrame, other_fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (team, date, competition) a team played — Premier League
    always; the other five competitions when `other_fixtures_df` is given
    (see `data.other_competitions.get_team_fixture_calendar`)."""
    pl_rows = to_team_perspective(matches_df)[["team", "date"]].copy()
    pl_rows["competition"] = "Premier League"
    if other_fixtures_df is None or other_fixtures_df.empty:
        return pl_rows
    extra = other_fixtures_df[["team", "date", "competition"]].copy()
    extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
    return pd.concat([pl_rows, extra], ignore_index=True)


def _counts_in_window(
    query_df: pd.DataFrame, calendar: pd.DataFrame, window_days: int
) -> pd.Series:
    """For each row of `query_df` (columns `team`, `date`), count `calendar`
    rows for that same team with a date in `[date - window_days, date)` —
    strictly before the query date, so a match never counts itself."""
    result = pd.Series(0, index=query_df.index, dtype=int)
    for team, group in query_df.groupby("team"):
        team_dates = np.sort(calendar.loc[calendar["team"] == team, "date"].to_numpy())
        query = group["date"].to_numpy()
        lo = np.searchsorted(team_dates, query - np.timedelta64(window_days, "D"), side="left")
        hi = np.searchsorted(team_dates, query, side="left")
        result.loc[group.index] = hi - lo
    return result


def build_congestion_features(
    matches_df: pd.DataFrame, other_fixtures_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Returns `games_last_14_days_home/away` and
    `european_fixture_last_4_days_home/away`, aligned to `matches_df`'s
    index. `other_fixtures_df` is optional — omitting it (or passing an
    empty frame) computes both purely from `matches_df` itself."""
    calendar = _full_calendar(matches_df, other_fixtures_df)
    european_calendar = calendar[calendar["competition"].isin(EUROPEAN_COMPETITIONS)]

    out = {}
    for side, team_col in (("home", "team_home"), ("away", "team_away")):
        query_df = pd.DataFrame({"team": matches_df[team_col], "date": matches_df["date"]}, index=matches_df.index)
        out[f"games_last_14_days_{side}"] = _counts_in_window(query_df, calendar, GAMES_WINDOW_DAYS)
        out[f"european_fixture_last_4_days_{side}"] = (
            _counts_in_window(query_df, european_calendar, EUROPEAN_WINDOW_DAYS) > 0
        ).astype(int)

    return pd.DataFrame(out, index=matches_df.index)
