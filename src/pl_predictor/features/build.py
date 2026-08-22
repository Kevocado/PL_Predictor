"""build.py — the single feature-construction entry point.

Every consumer (training, backtesting, the Streamlit app, and upcoming-
fixture prediction) calls `build_training_frame()` / `build_features_for`
rather than reimplementing feature logic — this is what keeps notebooks from
going stale the way FPL_Optimizer/exploration.ipynb did (it duplicated logic
inline instead of importing the shared module).
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data, understat
from . import cold_start, head_to_head, ratings, referee, rest_days, rolling_form, xg_form

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

    referee_feats = referee.build_referee_features(matches_df).reset_index(drop=True)

    understat_seasons = sorted({str(s)[:4] for s in matches_df["season"].unique()})
    xg_data = understat.load_xg_data(seasons=understat_seasons)
    xg_feats, xg_cols = xg_form.attach_xg_features(matches_df, xg_data)
    xg_feats = xg_feats.reset_index(drop=True)

    df = pd.concat(
        [df.reset_index(drop=True), elo_feats, pi_feats, h2h_feats, rest_feats, referee_feats, xg_feats], axis=1
    )
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["pi_diff"] = df["pi_home"] - df["pi_away"]

    xg_delta_feats, xg_delta_cols = xg_form.attach_xg_delta_features(df)
    df = pd.concat([df, xg_delta_feats], axis=1)

    feature_cols = (
        [f"home_{c}" for c in base_feature_cols]
        + [f"away_{c}" for c in base_feature_cols]
        + ["elo_home", "elo_away", "elo_diff", "pi_home", "pi_away", "pi_diff"]
        + ["h2h_home_goal_diff_avg", "h2h_home_win_rate"]
        + ["rest_days_home", "rest_days_away", "is_first_match_of_season_home", "is_first_match_of_season_away"]
        + [referee.FEATURE_COL]
        + xg_cols
        + xg_delta_cols
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


class FixtureFeatureContext:
    """Precomputes everything needed to build a live feature row for *any*
    (home, away) pair once per `matches_df` — the expensive part (Elo/Pi
    replay, every team's current rolling form, xG form) happens once here,
    so each individual fixture lookup afterward is cheap. Powers both
    `build_features_for_fixtures` (a batch of fixtures at once) and
    `models.ml_scoreline`'s live-serving wrapper (one pair at a time, on
    demand, from routes.py's already-cached matches_df) — one implementation
    of "build a live feature row," not two."""

    def __init__(self, matches_df: pd.DataFrame):
        self.matches_df = matches_df.sort_values("date").reset_index(drop=True)

        self.form = rolling_form.latest_form(self.matches_df)
        self.base_feature_cols = list(self.form.columns)
        self.league_avg = self.form.mean()
        self.games_played = self.matches_df.melt(value_vars=["team_home", "team_away"])["value"].value_counts()

        self.elo = ratings.fit_elo(self.matches_df)
        self.pi = ratings.fit_pi_ratings(self.matches_df)

        # Referee is never known this far ahead of an upcoming fixture (see
        # features/referee.py) — every live prediction uses the league average.
        self.referee_fallback = referee.league_average_card_rate(self.matches_df)

        understat_seasons = sorted({str(s)[:4] for s in self.matches_df["season"].unique()})
        xg_data = understat.load_xg_data(seasons=understat_seasons)
        self.xg_current = xg_form.latest_xg_form(xg_data)
        self.xg_league_avg = self.xg_current.mean() if not self.xg_current.empty else pd.Series(dtype=float)
        self.xg_stat_cols = {f"{stat}_last_{w}": w for stat in ("xg_for", "xg_against") for w in xg_form.WINDOWS}

    def build_row(self, home: str, away: str, commence_time=None) -> dict:
        row = {"team_home": home, "team_away": away, "commence_time": commence_time}

        for team, prefix in [(home, "home_"), (away, "away_")]:
            n_games = int(self.games_played.get(team, 0))
            team_form = self.form.loc[team] if team in self.form.index else pd.Series(dtype=float)
            # cold-start blend: weight real form by games_played / window, same idea as
            # cold_start.apply_cold_start_fallback but for a single "current" row.
            for col in self.base_feature_cols:
                w = cold_start._window_of(col) or 1
                weight = min(n_games / w, 1.0)
                avg = self.league_avg.get(col)
                current_val = team_form.get(col)
                if pd.isna(current_val):
                    current_val = avg
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}{col}"] = blended
            row[f"confidence_{prefix.rstrip('_')}"] = "current" if n_games >= max(rolling_form.LAG_WINDOWS) else (
                "blended" if n_games > 0 else "none"
            )

        row["elo_home"] = self.elo.get_team_rating(home)
        row["elo_away"] = self.elo.get_team_rating(away)
        row["elo_diff"] = row["elo_home"] - row["elo_away"]
        row["pi_home"] = self.pi.get_team_rating(home)
        row["pi_away"] = self.pi.get_team_rating(away)
        row["pi_diff"] = row["pi_home"] - row["pi_away"]

        row.update(_current_h2h(self.matches_df, home, away))

        fixture_date = commence_time or pd.Timestamp.now()
        home_rest = _current_rest_days(self.matches_df, home, fixture_date)
        away_rest = _current_rest_days(self.matches_df, away, fixture_date)
        row["rest_days_home"] = home_rest["rest_days"]
        row["rest_days_away"] = away_rest["rest_days"]
        row["is_first_match_of_season_home"] = home_rest["is_first_match_of_season"]
        row["is_first_match_of_season_away"] = away_rest["is_first_match_of_season"]

        row[referee.FEATURE_COL] = self.referee_fallback

        for team, prefix in [(home, "home"), (away, "away")]:
            n_games = int(self.games_played.get(team, 0))
            team_xg = self.xg_current.loc[team] if team in self.xg_current.index else pd.Series(dtype=float)
            for stat_col, w in self.xg_stat_cols.items():
                weight = min(n_games / w, 1.0)
                avg = self.xg_league_avg.get(stat_col)
                current_val = team_xg.get(stat_col)
                if pd.isna(current_val):
                    current_val = avg
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}_{stat_col}"] = blended

        for side in ("home", "away"):
            for w in xg_form.WINDOWS:
                row[f"{side}_xg_delta_for_last_{w}"] = row[f"{side}_last_{w}_goals_for"] - row[f"{side}_xg_for_last_{w}"]
                row[f"{side}_xg_delta_against_last_{w}"] = (
                    row[f"{side}_last_{w}_goals_against"] - row[f"{side}_xg_against_last_{w}"]
                )

        return row


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

    ctx = FixtureFeatureContext(matches_df)
    rows = [
        ctx.build_row(fixture["team_home"], fixture["team_away"], fixture.get("commence_time"))
        for _, fixture in fixtures_df.iterrows()
    ]
    return pd.DataFrame(rows)
