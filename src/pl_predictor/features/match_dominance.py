"""match_dominance.py — rolling match-dominance features from Understat's
per-shot detail (`data/understat_shots.py::load_match_dominance_data`).

Same shape as `xg_form.py`/`shot_situation.py`: shift(1).rolling windows per
team, joined onto the main matches frame via `merge_asof` (Understat has no
shared match identifier with football-data.co.uk). Research-only — not
imported by `features/build.py` until a walk-forward evaluation (isolated
per market, per the project's promotion gate) shows it earns a place in
`feature_cols`.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10)

_STATS = (
    "total_xg",
    "non_penalty_xg",
    "shots",
    "xg_per_shot",
    "open_play_xg_share",
    "set_piece_xg_share",
    "avg_shot_distance",
)


def _team_perspective(dominance_df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {"date": dominance_df["date"], "team": dominance_df["team_home"]}
        | {stat: dominance_df[f"home_{stat}"] for stat in _STATS}
    )
    away = pd.DataFrame(
        {"date": dominance_df["date"], "team": dominance_df["team_away"]}
        | {stat: dominance_df[f"away_{stat}"] for stat in _STATS}
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])


def build_rolling_dominance(dominance_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """Team-perspective long frame with rolling dominance columns — each
    row's value includes that team's own match, no `shift(1)`. Deliberate:
    `attach_dominance_features` joins this onto `matches_df` via
    `merge_asof(direction="backward", allow_exact_matches=False)`, which is
    what actually enforces "strictly before the target fixture" for a
    cross-source join like this one. Shifting here too would double up
    that exclusion and drop the source team's most recent match from every
    result — see `xg_form.py::build_rolling_xg`'s docstring for the
    confirmed real-data bug this exact mistake caused there."""
    long_df = _team_perspective(dominance_df)
    grouped = long_df.groupby("team", sort=False)

    new_cols = {}
    feature_cols = []
    for stat in _STATS:
        for w in windows:
            # "dominance_" prefix is deliberate: `set_piece_xg_share` would
            # otherwise collide with the existing (unused-in-feature_cols
            # but present-in-df) column of the same base name from
            # features/shot_situation.py's narrower, already-shipped
            # feature — namespacing every stat here avoids any ambiguity
            # about which module a column came from, not just this one.
            col = f"dominance_{stat}_last_{w}"
            new_cols[col] = grouped[stat].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            feature_cols.append(col)

    long_df = pd.concat([long_df, pd.DataFrame(new_cols, index=long_df.index)], axis=1)
    return long_df, feature_cols


def latest_dominance_form(dominance_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> pd.DataFrame:
    """Each team's current rolling dominance form (as of right now,
    including its most recent match) — for live-serving, analogous to
    `xg_form.latest_xg_form`. Indexed by team; empty if no dominance data
    was available."""
    if dominance_df.empty:
        return pd.DataFrame()

    long_df = _team_perspective(dominance_df)
    rows = {}
    for team, group in long_df.groupby("team"):
        row = {}
        for stat in _STATS:
            for w in windows:
                row[f"dominance_{stat}_last_{w}"] = group[stat].tail(w).mean()
        rows[team] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def attach_dominance_features(
    matches_df: pd.DataFrame, dominance_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS
) -> tuple[pd.DataFrame, list[str]]:
    """Left-joins each side's rolling dominance form onto `matches_df` (by
    team, as-of the match date) via `merge_asof`. Same contract as
    `xg_form.attach_xg_features`/`shot_situation.attach_shot_situation_
    features`: a frame aligned to `matches_df`'s row order, NaN where
    there's no coverage (older seasons Understat shot-level data hasn't
    been fetched for yet, or a scrape failure)."""
    if dominance_df.empty:
        cols = [f"{side}_{stat}_last_{w}" for side in ("home", "away") for stat in _STATS for w in windows]
        empty = pd.DataFrame(index=matches_df.index, columns=cols, dtype=float)
        return empty, cols

    long_df, base_cols = build_rolling_dominance(dominance_df, windows)

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
