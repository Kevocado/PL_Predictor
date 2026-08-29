"""covariate_poisson.py — a Dixon-Coles-*style* Poisson goal model that
accepts covariates, promoted from `evaluate/covariate_poisson_research.py`
after EXP-2026-15 (docs/AI_CONTINUITY.md): beat both Dixon-Coles and
Bivariate-Poisson on every walk-forward fold once given signed Elo/Pi
difference covariates on top of the classical team attack/defence
parameterization. penaltyblog's own DixonColesGoalModel/
BivariatePoissonGoalModel cannot accept covariates at all (confirmed by
inspecting their constructors) — this is a genuinely new model, not an
extension of either.

Team attack/defence strength: reshape each match into two rows (attacking
team, defending team, home indicator, goals scored), one-hot encode
attack/defence team, fit as a real Poisson GLM via
`sklearn.linear_model.PoissonRegressor` (already a project dependency via
scikit-learn) with the same `dixon_coles_weights(xi=0.0018)` recency
weighting Dixon-Coles/Bivariate-Poisson already use, so covariates sit
alongside team effects as ordinary regression coefficients.

Served only for markets `manifest.json`'s `scoreline.market_overrides`
names — see `models/scoreline.py::predict_fixture`'s `market_overrides`
parameter. It is not `chosen_model` for 1X2/overall scoreline serving;
`ml_scoreline` still wins that comparison.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.linear_model import PoissonRegressor

MIN_LAMBDA = 0.05
COVARIATE_COLS = ["elo_diff_signed", "pi_diff_signed"]


def _team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team-match-side: `attack_team`, `defence_team`,
    `is_home`, `goals_scored`, `date`, plus signed Elo/Pi-difference
    covariates (positive = this side stronger)."""
    home = pd.DataFrame(
        {
            "date": df["date"],
            "attack_team": df["team_home"],
            "defence_team": df["team_away"],
            "is_home": 1.0,
            "goals_scored": df["goals_home"],
            "elo_diff_signed": df["elo_home"] - df["elo_away"],
            "pi_diff_signed": df["pi_home"] - df["pi_away"],
        }
    )
    away = pd.DataFrame(
        {
            "date": df["date"],
            "attack_team": df["team_away"],
            "defence_team": df["team_home"],
            "is_home": 0.0,
            "goals_scored": df["goals_away"],
            "elo_diff_signed": df["elo_away"] - df["elo_home"],
            "pi_diff_signed": df["pi_away"] - df["pi_home"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _design_matrix(long_df: pd.DataFrame, team_categories: list[str]) -> pd.DataFrame:
    attack_cat = long_df["attack_team"].astype("category").cat.set_categories(team_categories)
    defence_cat = long_df["defence_team"].astype("category").cat.set_categories(team_categories)
    attack = pd.get_dummies(attack_cat, prefix="attack")
    defence = pd.get_dummies(defence_cat, prefix="defence")
    covariates = long_df[COVARIATE_COLS].fillna(0.0)
    X = pd.concat(
        [attack.reset_index(drop=True), defence.reset_index(drop=True), long_df[["is_home"]].reset_index(drop=True), covariates.reset_index(drop=True)],
        axis=1,
    )
    return X.astype(float)


def fit(train_df: pd.DataFrame, xi: float = 0.0018, alpha: float = 1.0, fit_rho: bool = False) -> "CovariatePoissonModel":
    """Fits on `train_df` (a `build_training_frame`-shaped historical
    frame — needs `elo_home`/`elo_away`/`pi_home`/`pi_away`, already
    computed there). Returns a `CovariatePoissonModel` with no live
    `.context` yet — set `.context` before calling `.predict(home, away)`
    for a genuine upcoming fixture (see `models/manifest.py::load_models`);
    `.predict_from_row` never needs it."""
    team_categories = sorted(set(train_df["team_home"]) | set(train_df["team_away"]))
    long_train = _team_perspective(train_df)
    X_train = _design_matrix(long_train, team_categories)

    weights = pb.models.dixon_coles_weights(long_train["date"], xi=xi)
    model = PoissonRegressor(alpha=alpha, max_iter=300)
    model.fit(X_train, long_train["goals_scored"], sample_weight=weights)
    result = CovariatePoissonModel(model, list(X_train.columns), team_categories)
    if fit_rho:
        result.rho = estimate_rho(result, train_df)
    return result


class CovariatePoissonModel:
    """Live-serving wrapper satisfying `models/scoreline.py`'s model
    contract: `.teams`, `.predict(home, away, max_goals)` for a genuine
    upcoming fixture (via `.context`'s current Elo/Pi ratings — same
    `features.build.FixtureFeatureContext` `MLScorelineModel` uses, shared
    rather than rebuilt — see `load_models`), and `.predict_from_row(row,
    max_goals)` for scoring an already-happened fixture from its own
    point-in-time `elo_home`/`elo_away`/`pi_home`/`pi_away` columns
    (leakage-safe, for calibration/backtest). Unlike `MLScorelineModel`,
    this has no `predict_many_from_rows` batched fast path — a
    `PoissonRegressor` predict call has none of XGBoost's per-call
    booster/DMatrix overhead, so the per-fixture loop
    `scoreline.predict_fixtures_batch` falls back to for any model without
    that method is already fast enough here; add one only if profiling
    ever shows otherwise."""

    def __init__(self, model: PoissonRegressor, columns: list[str], team_categories: list[str], context=None, rho: float = 0.0):
        self.model = model
        self.columns = columns
        self.teams = team_categories
        self.context = context
        self.rho = float(rho)

    def _lambda(self, attack_team: str, defence_team: str, is_home: bool, elo_diff: float, pi_diff: float) -> float:
        row = {c: 0.0 for c in self.columns}
        if f"attack_{attack_team}" in row:
            row[f"attack_{attack_team}"] = 1.0
        if f"defence_{defence_team}" in row:
            row[f"defence_{defence_team}"] = 1.0
        row["is_home"] = 1.0 if is_home else 0.0
        if "elo_diff_signed" in row:
            row["elo_diff_signed"] = elo_diff
        if "pi_diff_signed" in row:
            row["pi_diff_signed"] = pi_diff
        X = pd.DataFrame([row], columns=self.columns)
        return max(float(self.model.predict(X)[0]), MIN_LAMBDA)

    def predict(self, home: str, away: str, max_goals: int = 10, **_kwargs):
        if self.context is None:
            raise RuntimeError("CovariatePoissonModel.context must be set before predicting a live fixture")
        elo_home, elo_away = self.context.elo.get_team_rating(home), self.context.elo.get_team_rating(away)
        pi_home, pi_away = self.context.pi.get_team_rating(home), self.context.pi.get_team_rating(away)
        lam_home = self._lambda(home, away, True, elo_home - elo_away, pi_home - pi_away)
        lam_away = self._lambda(away, home, False, elo_away - elo_home, pi_away - pi_home)
        return pb.models.create_dixon_coles_grid(lam_home, lam_away, rho=self.rho, max_goals=max_goals)

    def predict_from_row(self, row, max_goals: int = 10):
        lam_home = self._lambda(row["team_home"], row["team_away"], True, row["elo_home"] - row["elo_away"], row["pi_home"] - row["pi_away"])
        lam_away = self._lambda(row["team_away"], row["team_home"], False, row["elo_away"] - row["elo_home"], row["pi_away"] - row["pi_home"])
        return pb.models.create_dixon_coles_grid(lam_home, lam_away, rho=self.rho, max_goals=max_goals)


def predict_grids_batch(model: CovariatePoissonModel, val_df: pd.DataFrame, max_goals: int = 10) -> list:
    """Vectorized scoring of an already-built frame's own point-in-time
    columns (the `predict_from_row` case, batched) — used for evaluation,
    where scoring hundreds of historical fixtures one Python call at a
    time would be needlessly slow even without XGBoost-style overhead."""
    long_val = _team_perspective(val_df)
    X_val = _design_matrix(long_val, model.teams).reindex(columns=model.columns, fill_value=0.0)
    preds = np.maximum(model.model.predict(X_val), MIN_LAMBDA)

    n = len(val_df)
    lam_home, lam_away = preds[:n], preds[n:]
    return [
        pb.models.create_dixon_coles_grid(float(h), float(a), rho=model.rho, max_goals=max_goals)
        for h, a in zip(lam_home, lam_away)
    ]


def estimate_rho(model: CovariatePoissonModel, train_df: pd.DataFrame, candidates: np.ndarray | None = None) -> float:
    """Select a Dixon-Coles correlation on training likelihood only.

    Validation folds are never used to tune rho.  The promotion decision is
    made separately by ``evaluate.covariate_poisson_research`` across every
    latest-season fold and calibration metric.
    """
    candidates = candidates if candidates is not None else np.linspace(-0.15, 0.15, 13)
    long_df = _team_perspective(train_df)
    X = _design_matrix(long_df, model.teams).reindex(columns=model.columns, fill_value=0.0)
    lambdas = np.maximum(model.model.predict(X), MIN_LAMBDA)
    n = len(train_df)
    home, away = lambdas[:n], lambdas[n:]
    goals_home = train_df["goals_home"].astype(int).to_numpy()
    goals_away = train_df["goals_away"].astype(int).to_numpy()
    best_rho, best_loss = 0.0, float("inf")
    for rho in candidates:
        losses = []
        for h, a, gh, ga in zip(home, away, goals_home, goals_away):
            grid = pb.models.create_dixon_coles_grid(float(h), float(a), rho=float(rho), max_goals=max(10, gh + 2, ga + 2))
            losses.append(-np.log(max(float(grid.exact_score(int(gh), int(ga))), 1e-12)))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_rho, best_loss = float(rho), loss
    return best_rho


def save(model: CovariatePoissonModel, path: Path) -> None:
    """`.context` is deliberately never pickled — it's a `FixtureFeatureContext`
    built from current match data, always reattached at load time (see
    `models/manifest.py::load_models`), not something to freeze at train time."""
    with open(path, "wb") as f:
        pickle.dump({"model": model.model, "columns": model.columns, "teams": model.teams, "rho": model.rho}, f)


def load(path: Path) -> CovariatePoissonModel:
    with open(path, "rb") as f:
        state = pickle.load(f)
    return CovariatePoissonModel(state["model"], state["columns"], state["teams"], rho=state.get("rho", 0.0))
