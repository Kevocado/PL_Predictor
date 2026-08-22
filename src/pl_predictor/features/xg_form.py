"""xg_form.py — rolling team-level expected-goals (xG) form.

Same shift(1).rolling(window) idiom as team_names and rolling_form.py, just
on Understat's xG_for/xG_against instead of actual goals/shots/corners.
Joined onto the main matches frame via `merge_asof` rather than an exact
fixture-id match, since Understat and football-data.co.uk are independent
scrapers with no shared match identifier — "this team's most recent known
rolling xG as of just before this date" is well-defined and robust without
needing row-for-row reconciliation between the two sources.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10)


def _team_perspective(understat_df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": understat_df["date"],
            "team": understat_df["team_home"],
            "xg_for": understat_df["xg_home"],
            "xg_against": understat_df["xg_away"],
        }
    )
    away = pd.DataFrame(
        {
            "date": understat_df["date"],
            "team": understat_df["team_away"],
            "xg_for": understat_df["xg_away"],
            "xg_against": understat_df["xg_home"],
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])


def build_rolling_xg(understat_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """Team-perspective long frame with rolling xG-for/xG-against columns,
    using only *prior* matches (shift(1))."""
    long_df = _team_perspective(understat_df)
    grouped = long_df.groupby("team", sort=False)

    feature_cols = []
    for w in windows:
        for stat in ("xg_for", "xg_against"):
            col = f"{stat}_last_{w}"
            long_df[col] = grouped[stat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            feature_cols.append(col)

    return long_df, feature_cols


def latest_xg_form(understat_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> pd.DataFrame:
    """Each team's *current* rolling xG form (as of right now, including
    their most recent match) — for scoring an upcoming fixture. Analogous to
    `rolling_form.latest_form`. Indexed by team; empty if Understat data
    wasn't available."""
    if understat_df.empty:
        return pd.DataFrame()

    long_df = _team_perspective(understat_df)
    rows = {}
    for team, group in long_df.groupby("team"):
        row = {}
        for w in windows:
            row[f"xg_for_last_{w}"] = group["xg_for"].tail(w).mean()
            row[f"xg_against_last_{w}"] = group["xg_against"].tail(w).mean()
        rows[team] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def attach_xg_delta_features(df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """Actual rolling goals minus rolling xG, for/against, per side/window —
    a regression-to-mean signal (a team scoring well above its underlying xG
    recently tends to cool off, and vice versa) distinct from either rolling
    goals or rolling xG alone. Must be called after both `home_last_{w}_
    goals_for/against` (rolling_form) and `{side}_xg_for/against_last_{w}`
    (attach_xg_features) are already columns on `df` — e.g. right before
    `build.build_training_frame` assembles its final `feature_cols` list, or
    from `FixtureFeatureContext.build_row` once its own two feature loops
    have populated `row` with both of those. Validated (see
    market_odds_experiment scratchpad, 2026-08-22): closes a small further
    slice of the RPS/Brier gap to the bookmaker beyond rolling goals/xG
    alone (RPS 0.2081->0.2073, Brier 0.6169->0.6159 on the 2025-26 holdout)."""
    cols = {}
    feature_cols = []
    for side in ("home", "away"):
        for w in windows:
            delta_for, delta_against = f"{side}_xg_delta_for_last_{w}", f"{side}_xg_delta_against_last_{w}"
            cols[delta_for] = df[f"{side}_last_{w}_goals_for"] - df[f"{side}_xg_for_last_{w}"]
            cols[delta_against] = df[f"{side}_last_{w}_goals_against"] - df[f"{side}_xg_against_last_{w}"]
            feature_cols += [delta_for, delta_against]
    return pd.DataFrame(cols, index=df.index), feature_cols


def attach_xg_features(matches_df: pd.DataFrame, understat_df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """Left-joins each side's rolling xG form onto `matches_df` (by team,
    as-of the match date) via `merge_asof`. Returns a frame aligned to
    `matches_df`'s row order with `home_xg_*`/`away_xg_*` columns, plus the
    feature-column list. Rows with no Understat coverage (older seasons, or
    if the scrape failed) get NaN — callers should treat these the same as
    any other missing feature (e.g. `.fillna(0)` before feeding a model)."""
    if understat_df.empty:
        cols = [f"{side}_xg_{stat}_last_{w}" for side in ("home", "away") for stat in ("for", "against") for w in windows]
        empty = pd.DataFrame(index=matches_df.index, columns=cols, dtype=float)
        return empty, cols

    long_df, base_cols = build_rolling_xg(understat_df, windows)

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

    home_xg = _asof_join("team_home", "home")
    away_xg = _asof_join("team_away", "away")

    out = pd.concat([home_xg, away_xg], axis=1)
    return out, list(out.columns)
