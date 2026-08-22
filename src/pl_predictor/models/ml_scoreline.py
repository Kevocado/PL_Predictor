"""ml_scoreline.py — feeds engineered features (rolling form, Elo/Pi,
referee, xG) into XGBoost expected-goals regressors, then prices markets via
penaltyblog's `create_dixon_coles_grid` — evaluated on held-out data, and
(when it wins) usable as a genuine `models/manifest.py::chosen_model`
alongside Dixon-Coles/Bivariate-Poisson.

`predict_fixture(model, home, away)` — used everywhere in the app
(routes.py, player_goals.py, projected_table.py) — assumes a model that's
self-sufficient given just two team names. That's true for Dixon-Coles/
Bivariate-Poisson (team-level parameters fit directly from goals), but an ML
model needs a live *feature vector* for that specific fixture. `MLScorelineModel`
below closes that gap: it holds a `features.build.FixtureFeatureContext`
(the same one `build_features_for_fixtures` uses, built once from the
current `matches_df`) and builds each fixture's row on demand inside
`.predict()` — so it satisfies the exact same duck-typed interface
(`.teams`, `.predict(home, away, max_goals)`) as the penaltyblog models, and
plugs into every existing call site with no changes there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
import xgboost as xgb

MIN_LAMBDA = 0.05  # create_dixon_coles_grid requires strictly positive lambdas


def _regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_goal_regressors(X_train, goals_home_train, goals_away_train) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    home_model = _regressor()
    home_model.fit(X_train, goals_home_train)
    away_model = _regressor()
    away_model.fit(X_train, goals_away_train)
    return home_model, away_model


def predict_grid(home_model: xgb.XGBRegressor, away_model: xgb.XGBRegressor, x_row: pd.DataFrame, max_goals: int = 10):
    lam_home = max(float(home_model.predict(x_row)[0]), MIN_LAMBDA)
    lam_away = max(float(away_model.predict(x_row)[0]), MIN_LAMBDA)
    return pb.models.create_dixon_coles_grid(lam_home, lam_away, rho=0.0, max_goals=max_goals)


def evaluate_on_holdout(home_model, away_model, X_val: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    result_code = {"H": 0, "D": 1, "A": 2}
    probs = []
    for i in range(len(X_val)):
        grid = predict_grid(home_model, away_model, X_val.iloc[[i]])
        probs.append([grid.home_win, grid.draw, grid.away_win])
    probs = np.array(probs)
    outcomes = val_df["ftr"].map(result_code).to_numpy()
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
    }


class MLScorelineModel:
    """Live-serving wrapper: satisfies the same `.teams` + `.predict(home,
    away, max_goals)` interface as penaltyblog's goal models, so it's a
    drop-in `models/manifest.py::chosen_model` option. `context` is a
    `features.build.FixtureFeatureContext` built from the current
    `matches_df` — expensive parts (Elo/Pi replay, every team's rolling
    form) already done once at construction, so `.predict()` itself is
    cheap."""

    def __init__(self, home_model, away_model, feature_cols: list[str], teams: list[str], context):
        self.home_model = home_model
        self.away_model = away_model
        self.feature_cols = feature_cols
        self.teams = teams
        self.context = context

    def predict(self, home: str, away: str, max_goals: int = 10, **_kwargs):
        row = self.context.build_row(home, away)
        # A single-row frame built from a dict can leave an all-None column
        # (e.g. h2h_* for a pair with no prior meetings) as object dtype even
        # after fillna(0) — XGBoost's inplace_predict rejects object dtypes
        # outright, so force numeric here rather than relying on fillna alone.
        x = pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0).fillna(0).astype(float)
        return predict_grid(self.home_model, self.away_model, x, max_goals=max_goals)

    def predict_from_row(self, row, max_goals: int = 10):
        """Score a fixture using its own precomputed, point-in-time feature
        columns (e.g. a row of `build_training_frame`'s output) instead of
        `context`'s "current state as of today" lookup. This is the only
        leakage-safe way to evaluate a fixture that's already in the
        historical record — `.predict(home, away)` would otherwise pull in
        each team's rolling form/Elo/xG as of *now*, which for a past match
        includes results that happened after it (and, for anything in the
        held-out season, effectively the season's own outcome). Genuine
        upcoming fixtures have no such point-in-time features to fall back
        on, so they correctly keep using `.predict(home, away)` instead."""
        x = pd.DataFrame([row])[self.feature_cols].fillna(0).astype(float)
        return predict_grid(self.home_model, self.away_model, x, max_goals=max_goals)
