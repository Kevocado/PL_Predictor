"""shot_situation.py — rolling set-piece xG share.

Same shape as `xg_form.py` (rolling window + `merge_asof` join by date,
since `data/understat_shots.py` is an independently-scheduled source with
no shared match identifier against `matches_df`), just for one stat instead
of two: what share of a team's xG recently has come from a dead-ball
restart (corner, free kick, penalty) rather than open play — a real style
signal a single xG number doesn't distinguish. Real coverage gaps match
Understat's own limits already documented in `data/understat.py` (reliable
from the 2014-15 season on) — same NaN-safe handling as the existing xG
features.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10)


def _team_perspective(shot_df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": shot_df["date"],
            "team": shot_df["team_home"],
            "set_piece_xg_share": shot_df["home_set_piece_xg_share"],
        }
    )
    away = pd.DataFrame(
        {
            "date": shot_df["date"],
            "team": shot_df["team_away"],
            "set_piece_xg_share": shot_df["away_set_piece_xg_share"],
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])


def build_rolling_shot_situation(shot_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    long_df = _team_perspective(shot_df)
    grouped = long_df.groupby("team", sort=False)

    feature_cols = []
    for w in windows:
        col = f"set_piece_xg_share_last_{w}"
        long_df[col] = grouped["set_piece_xg_share"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
        feature_cols.append(col)

    return long_df, feature_cols


def latest_shot_situation_form(shot_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> pd.DataFrame:
    """Each team's current rolling set-piece xG share (as of right now) —
    for live-serving, analogous to `xg_form.latest_xg_form`. Indexed by
    team; empty if no shot-situation data was available."""
    if shot_df.empty:
        return pd.DataFrame()

    long_df = _team_perspective(shot_df)
    rows = {}
    for team, group in long_df.groupby("team"):
        row = {}
        for w in windows:
            row[f"set_piece_xg_share_last_{w}"] = group["set_piece_xg_share"].tail(w).mean()
        rows[team] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def attach_shot_situation_features(
    matches_df: pd.DataFrame, shot_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS
) -> tuple[pd.DataFrame, list[str]]:
    """Left-joins each side's rolling set-piece xG share onto `matches_df`
    (by team, as-of the match date) via `merge_asof`. Same contract as
    `xg_form.attach_xg_features`: a frame aligned to `matches_df`'s row
    order, NaN where there's no coverage."""
    if shot_df.empty:
        cols = [f"{side}_set_piece_xg_share_last_{w}" for side in ("home", "away") for w in windows]
        empty = pd.DataFrame(index=matches_df.index, columns=cols, dtype=float)
        return empty, cols

    long_df, base_cols = build_rolling_shot_situation(shot_df, windows)

    left = matches_df[["date", "team_home", "team_away"]].reset_index().sort_values("date")

    def _asof_join(team_col: str, prefix: str) -> pd.DataFrame:
        right = long_df.rename(columns={"team": team_col}).sort_values("date")
        merged = pd.merge_asof(
            left[["index", "date", team_col]].sort_values("date"),
            right[["date", team_col] + base_cols],
            on="date",
            by=team_col,
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged.set_index("index").reindex(matches_df.index)
        return merged[base_cols].rename(columns={c: f"{prefix}_{c}" for c in base_cols})

    home = _asof_join("team_home", "home")
    away = _asof_join("team_away", "away")

    out = pd.concat([home, away], axis=1)
    return out, list(out.columns)
