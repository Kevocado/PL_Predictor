"""build.py — the single feature-construction entry point.

Every consumer (training, backtesting, the Streamlit app, and upcoming-
fixture prediction) calls `build_training_frame()` / `build_features_for`
rather than reimplementing feature logic — this is what keeps notebooks from
going stale the way FPL_Optimizer/exploration.ipynb did (it duplicated logic
inline instead of importing the shared module).
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data
from . import cold_start, head_to_head, ratings, rest_days, rolling_form

# Targets / raw-outcome columns that must never appear in the feature list
# (that would be leaking the match's own result into its own features).
TARGET_COLS = ["goals_home", "goals_away", "ftr", "total_corners", "total_cards"]


def _add_targets(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()
    zeros = pd.Series(0, index=df.index)
    df["total_corners"] = df.get("hc", zeros).fillna(0) + df.get("ac", zeros).fillna(0)
    df["total_cards"] = (
        df.get("hy", zeros).fillna(0)
        + df.get("ay", zeros).fillna(0)
        + df.get("hr", zeros).fillna(0)
        + df.get("ar", zeros).fillna(0)
    )
    return df


def build_training_frame(
    seasons: list[str] | None = None, matches_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (df, feature_cols) — one row per historical match, with
    targets (goals_home/away, ftr, total_corners, total_cards) plus all
    model-ready feature columns (rolling form, ratings, h2h, rest days).
    `feature_cols` is the canonical list to pass to any model."""
    if matches_df is None:
        matches_df = football_data.load_training_data(seasons=seasons)
    matches_df = _add_targets(matches_df)

    long_df, base_feature_cols = rolling_form.build_rolling_form(matches_df)
    long_df, confidence = cold_start.apply_cold_start_fallback(long_df, base_feature_cols)
    long_df["confidence"] = confidence

    home_join = (
        long_df[long_df["was_home"]]
        .set_index(["date", "team"])[base_feature_cols + ["confidence"]]
        .rename(columns={c: f"home_{c}" for c in base_feature_cols})
        .rename(columns={"confidence": "confidence_home"})
        .reset_index()
        .rename(columns={"team": "team_home"})
    )
    away_join = (
        long_df[~long_df["was_home"]]
        .set_index(["date", "team"])[base_feature_cols + ["confidence"]]
        .rename(columns={c: f"away_{c}" for c in base_feature_cols})
        .rename(columns={"confidence": "confidence_away"})
        .reset_index()
        .rename(columns={"team": "team_away"})
    )

    df = matches_df.merge(home_join, on=["date", "team_home"], how="left")
    df = df.merge(away_join, on=["date", "team_away"], how="left")

    elo_feats = ratings.replay_elo(matches_df).reset_index(drop=True)
    pi_feats = ratings.replay_pi_ratings(matches_df).reset_index(drop=True)
    h2h_feats = head_to_head.build_h2h_features(matches_df).reset_index(drop=True)
    rest_feats = rest_days.build_rest_days(matches_df).reset_index(drop=True)

    df = pd.concat(
        [df.reset_index(drop=True), elo_feats, pi_feats, h2h_feats, rest_feats], axis=1
    )
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["pi_diff"] = df["pi_home"] - df["pi_away"]

    feature_cols = (
        [f"home_{c}" for c in base_feature_cols]
        + [f"away_{c}" for c in base_feature_cols]
        + ["elo_home", "elo_away", "elo_diff", "pi_home", "pi_away", "pi_diff"]
        + ["h2h_home_goal_diff_avg", "h2h_home_win_rate"]
        + ["rest_days_home", "rest_days_away", "is_first_match_of_season_home", "is_first_match_of_season_away"]
    )

    return df, feature_cols


def _current_h2h(matches_df: pd.DataFrame, home: str, away: str, window: int = head_to_head.H2H_WINDOW) -> dict:
    pair = head_to_head._pair_key(home, away)
    df = matches_df.copy()
    df["pair"] = [head_to_head._pair_key(h, a) for h, a in zip(df["team_home"], df["team_away"])]
    past = df[df["pair"] == pair].sort_values("date").tail(window)
    if past.empty:
        return {"h2h_home_goal_diff_avg": None, "h2h_home_win_rate": None}

    diffs = [
        (gd if h == home else -gd)
        for h, gd in zip(past["team_home"], past["goals_home"] - past["goals_away"])
    ]
    return {
        "h2h_home_goal_diff_avg": sum(diffs) / len(diffs),
        "h2h_home_win_rate": sum(1 for d in diffs if d > 0) / len(diffs),
    }


def _current_rest_days(matches_df: pd.DataFrame, team: str, as_of_date) -> dict:
    team_matches = matches_df[(matches_df["team_home"] == team) | (matches_df["team_away"] == team)]
    if team_matches.empty:
        return {"rest_days": None, "is_first_match_of_season": True}
    last_date = team_matches["date"].max()
    same_season = (
        team_matches[team_matches["date"] == last_date]["season"].iloc[0] == matches_df["season"].iloc[-1]
    )
    return {
        "rest_days": (pd.Timestamp(as_of_date) - last_date).days,
        "is_first_match_of_season": not same_season,
    }


def build_features_for_fixtures(
    fixtures_df: pd.DataFrame, matches_df: pd.DataFrame | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """Feature rows for upcoming (not-yet-played) fixtures, using each team's
    *current* state (latest rolling form / ratings / h2h / rest days) rather
    than the shift(1) historical features `build_training_frame` produces.
    `fixtures_df` needs `team_home`, `team_away`, `commence_time` columns
    (see `data.fixtures.get_upcoming_fixtures`)."""
    if matches_df is None:
        matches_df = football_data.load_training_data(seasons=seasons)
        current = football_data.fetch_current_season_partial()
        if current is not None and not current.empty:
            matches_df = pd.concat([matches_df, current], ignore_index=True)
    matches_df = matches_df.sort_values("date").reset_index(drop=True)

    form = rolling_form.latest_form(matches_df)
    base_feature_cols = [c for c in form.columns]
    league_avg = form.mean()
    games_played = matches_df.melt(value_vars=["team_home", "team_away"])["value"].value_counts()

    elo = ratings.fit_elo(matches_df)
    pi = ratings.fit_pi_ratings(matches_df)

    rows = []
    for _, fixture in fixtures_df.iterrows():
        home, away = fixture["team_home"], fixture["team_away"]
        row = {"team_home": home, "team_away": away, "commence_time": fixture.get("commence_time")}

        for team, prefix in [(home, "home_"), (away, "away_")]:
            n_games = int(games_played.get(team, 0))
            team_form = form.loc[team] if team in form.index else pd.Series(dtype=float)
            # cold-start blend: weight real form by games_played / window, same idea as
            # cold_start.apply_cold_start_fallback but for a single "current" row.
            for col in base_feature_cols:
                w = cold_start._window_of(col) or 1
                weight = min(n_games / w, 1.0)
                avg = league_avg.get(col)
                current_val = team_form.get(col)
                if pd.isna(current_val):
                    current_val = avg
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}{col}"] = blended
            row[f"confidence_{prefix.rstrip('_')}"] = "current" if n_games >= max(rolling_form.LAG_WINDOWS) else (
                "blended" if n_games > 0 else "none"
            )

        row["elo_home"] = elo.get_team_rating(home)
        row["elo_away"] = elo.get_team_rating(away)
        row["elo_diff"] = row["elo_home"] - row["elo_away"]
        row["pi_home"] = pi.get_team_rating(home)
        row["pi_away"] = pi.get_team_rating(away)
        row["pi_diff"] = row["pi_home"] - row["pi_away"]

        row.update(_current_h2h(matches_df, home, away))

        fixture_date = fixture.get("commence_time") or pd.Timestamp.now()
        home_rest = _current_rest_days(matches_df, home, fixture_date)
        away_rest = _current_rest_days(matches_df, away, fixture_date)
        row["rest_days_home"] = home_rest["rest_days"]
        row["rest_days_away"] = away_rest["rest_days"]
        row["is_first_match_of_season_home"] = home_rest["is_first_match_of_season"]
        row["is_first_match_of_season_away"] = away_rest["is_first_match_of_season"]

        rows.append(row)

    return pd.DataFrame(rows)
