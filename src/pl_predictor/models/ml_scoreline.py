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

# `DEFAULT_HYPERPARAMS` is the single source of truth both `_regressor()`
# and `evaluate/tune_hyperparams.py` (the Optuna search space) start from.
#
# Tuned via `evaluate/tune_hyperparams.py` (50-trial Optuna/TPE search,
# objective = mean RPS across `evaluate/walk_forward.py`'s 5 rolling
# validation folds, not the single fixed holdout — chosen specifically to
# avoid overfitting the hyperparameters to one season's quirks). Measured
# directly: walk-forward mean RPS 0.20128->0.19955 (-0.86%), AND — checked
# separately, since a walk-forward win alone wouldn't rule out overfitting
# those specific folds — the single fixed chronological-holdout RPS/Brier
# also improved (0.20736->0.20690 RPS, 0.61572->0.61403 Brier), so this is
# a corroborated win, not a walk-forward-specific artifact. The dominant
# change is much stronger L2 regularization (reg_lambda 1.0->18.1) — this
# was expected: this session found repeatedly (fouls in the shared set,
# table-context, shot-situation) that adding columns to this feature set
# hurt via overfitting on ~2,280 rows; substantially more L2 shrinkage is
# a direct, corroborated fix for exactly that, not a new hypothesis.
#
# Previous defaults (kept here for reference / a quick revert if a future
# retrain regresses unexpectedly): n_estimators=300, learning_rate=0.05,
# max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
# reg_lambda=1.0 (XGBoost's own internal default when unset — confirmed via
# a fitted booster's saved config; the sklearn wrapper reports "None" for
# unset, which does NOT mean zero, a mistake this project's own earlier
# research made and caught here via the same before/after discipline used
# throughout).
DEFAULT_HYPERPARAMS: dict = {
    "n_estimators": 250,
    "learning_rate": 0.0232,
    "max_depth": 4,
    "subsample": 0.5705,
    "colsample_bytree": 0.9011,
    "reg_alpha": 0.002,
    "reg_lambda": 18.1027,
}


def _regressor(hyperparams: dict | None = None) -> xgb.XGBRegressor:
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    return xgb.XGBRegressor(objective="count:poisson", random_state=42, **params)


def train_goal_regressors(
    X_train,
    goals_home_train,
    goals_away_train,
    dates: pd.Series | None = None,
    xi: float = 0.0018,
    hyperparams: dict | None = None,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """`dates` (each row's match date, same length/order as `X_train`) is
    optional but should always be passed in production: without it every
    match counts equally regardless of age, unlike Dixon-Coles/Bivariate-
    Poisson which already downweight older matches via the same
    `dixon_coles_weights(xi=0.0018)` used here — omitting it would let a
    growing pile of old, equally-weighted matches dilute recent signal
    (playing styles, squad economics) as the training window widens.

    `hyperparams` overrides any subset of `DEFAULT_HYPERPARAMS` (e.g. from
    an Optuna trial) — omit for production defaults."""
    sample_weight = None
    if dates is not None:
        sample_weight = np.asarray(pb.models.dixon_coles_weights(dates, xi=xi), dtype=np.float64)
    home_model = _regressor(hyperparams)
    home_model.fit(X_train, goals_home_train, sample_weight=sample_weight)
    away_model = _regressor(hyperparams)
    away_model.fit(X_train, goals_away_train, sample_weight=sample_weight)
    return home_model, away_model


def predict_grid(home_model: xgb.XGBRegressor, away_model: xgb.XGBRegressor, x_row: pd.DataFrame, max_goals: int = 10):
    lam_home = max(float(home_model.predict(x_row)[0]), MIN_LAMBDA)
    lam_away = max(float(away_model.predict(x_row)[0]), MIN_LAMBDA)
    return pb.models.create_dixon_coles_grid(lam_home, lam_away, rho=0.0, max_goals=max_goals)


def predict_grids_batch(home_model: xgb.XGBRegressor, away_model: xgb.XGBRegressor, X: pd.DataFrame, max_goals: int = 10) -> list:
    """Same result as calling `predict_grid` once per row, but one XGBoost
    `.predict()` call for the whole batch instead of one call per row.
    XGBoost has fixed per-call overhead (booster/DMatrix setup) that's
    negligible for one prediction but dominates when looped — measured at
    ~5s for 380 rows looped vs a small fraction of a second batched. Always
    prefer this over a Python loop of `predict_grid` calls whenever more
    than a handful of fixtures need scoring at once (holdout evaluation,
    live calibration, backtesting)."""
    lam_home = np.maximum(home_model.predict(X), MIN_LAMBDA)
    lam_away = np.maximum(away_model.predict(X), MIN_LAMBDA)
    return [
        pb.models.create_dixon_coles_grid(float(h), float(a), rho=0.0, max_goals=max_goals)
        for h, a in zip(lam_home, lam_away)
    ]


def evaluate_on_holdout(home_model, away_model, X_val: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    result_code = {"H": 0, "D": 1, "A": 2}
    grids = predict_grids_batch(home_model, away_model, X_val)
    probs = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
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

    def predict_many_from_rows(self, df: pd.DataFrame, max_goals: int = 10) -> list:
        """Batched `predict_from_row` — see `predict_grids_batch`'s
        docstring for why this matters. Used wherever a whole historical
        frame (calibration, backtest) needs scoring at once instead of
        fixture-by-fixture."""
        X = df[self.feature_cols].fillna(0).astype(float)
        return predict_grids_batch(self.home_model, self.away_model, X, max_goals=max_goals)
