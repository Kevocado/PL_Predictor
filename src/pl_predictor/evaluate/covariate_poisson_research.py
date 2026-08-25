"""covariate_poisson_research.py — Part 3b of the match-dominance plan: a
Dixon-Coles-*style* Poisson goal model that can accept covariates, since
penaltyblog's own DixonColesGoalModel/BivariatePoissonGoalModel cannot
(confirmed by direct inspection of their constructors — only
goals/teams/weights/neutral_venue, no covariate slot at all).

Team attack/defence strength is represented the classical way for this
family of model: reshape each match into two rows (attacking team,
defending team, home indicator, goals scored — see `_team_perspective`),
one-hot encode attack/defence team, and fit a genuine Poisson GLM
(`sklearn.linear_model.PoissonRegressor`, already a project dependency via
scikit-learn — no new dependency needed) with the same
`dixon_coles_weights(xi=0.0018)` recency weighting DC/BP already use.
External features (Elo/Pi difference, the match-dominance rolling stats)
sit alongside the team dummies as ordinary regression coefficients.

Research-only: this is not `models/covariate_poisson.py` because it has
not earned production status. Promote it there only if a chronological/
walk-forward comparison shows it beats Dixon-Coles/Bivariate-Poisson (the
two models it's structurally closest to) by enough to matter, corroborated
on the most recent season per this project's established bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.linear_model import PoissonRegressor

from ..models import scoreline
from .model_selection_by_segment import prepare_folds
from .scoreline_dominance_arms import _metrics_from_grids

MIN_LAMBDA = 0.05


def _team_perspective(df: pd.DataFrame, dominance_extra_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per team-match-side: `attack_team`, `defence_team`,
    `is_home`, `goals_scored`, `date`, plus signed Elo/Pi-difference
    covariates (positive = this side stronger) and, if
    `dominance_extra_cols` is given, this side's own rolling
    match-dominance stats (never the opponent's — this is each team's own
    attacking/defensive profile, not a relative comparison, since the
    dummy variables already capture relative team strength)."""
    dominance_extra_cols = dominance_extra_cols or []
    home_dominance_cols = [c for c in dominance_extra_cols if c.startswith("home_")]
    away_dominance_cols = [c for c in dominance_extra_cols if c.startswith("away_")]
    dominance_bases = [c[len("home_") :] for c in home_dominance_cols]

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
    for base, col in zip(dominance_bases, home_dominance_cols):
        home[f"own_{base}"] = df[col]

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
    for base, col in zip(dominance_bases, away_dominance_cols):
        away[f"own_{base}"] = df[col]

    return pd.concat([home, away], ignore_index=True)


def _design_matrix(long_df: pd.DataFrame, covariate_cols: list[str], team_categories: list[str]) -> pd.DataFrame:
    # .astype("category").cat.set_categories(...) rather than
    # pd.Categorical(values, categories=...) directly: the latter is
    # deprecated by pandas for values containing entries outside
    # `categories` (exactly the case here — a val-season team never seen
    # in train_df) even though the resulting behavior (NaN category -> an
    # all-zero dummy row, the intended graceful fallback) is unchanged.
    attack_cat = long_df["attack_team"].astype("category").cat.set_categories(team_categories)
    defence_cat = long_df["defence_team"].astype("category").cat.set_categories(team_categories)
    attack = pd.get_dummies(attack_cat, prefix="attack")
    defence = pd.get_dummies(defence_cat, prefix="defence")
    covariates = long_df[covariate_cols].fillna(0.0) if covariate_cols else pd.DataFrame(index=long_df.index)
    X = pd.concat(
        [attack.reset_index(drop=True), defence.reset_index(drop=True), long_df[["is_home"]].reset_index(drop=True), covariates.reset_index(drop=True)],
        axis=1,
    )
    return X.astype(float)


def fit(
    train_df: pd.DataFrame,
    covariate_cols: list[str],
    dominance_extra_cols: list[str] | None = None,
    xi: float = 0.0018,
    alpha: float = 1.0,
) -> tuple[PoissonRegressor, list[str], list[str]]:
    """Fits one covariate-Poisson model on `train_df` (a `build_training_
    frame`-shaped historical frame). Returns (model, design_matrix_columns,
    team_categories) — both needed to build a compatible design matrix for
    prediction later, since an unseen team at validation time must fall
    back to all-zero attack/defence dummies (letting the intercept, home
    indicator, and covariates alone determine its rate) rather than error."""
    team_categories = sorted(set(train_df["team_home"]) | set(train_df["team_away"]))
    long_train = _team_perspective(train_df, dominance_extra_cols)
    X_train = _design_matrix(long_train, covariate_cols, team_categories)

    weights = pb.models.dixon_coles_weights(long_train["date"], xi=xi)
    model = PoissonRegressor(alpha=alpha, max_iter=300)
    model.fit(X_train, long_train["goals_scored"], sample_weight=weights)
    return model, list(X_train.columns), team_categories


def predict_grids_batch(
    model: PoissonRegressor,
    columns: list[str],
    team_categories: list[str],
    val_df: pd.DataFrame,
    covariate_cols: list[str],
    dominance_extra_cols: list[str] | None = None,
    max_goals: int = 10,
) -> list:
    long_val = _team_perspective(val_df, dominance_extra_cols)
    X_val = _design_matrix(long_val, covariate_cols, team_categories).reindex(columns=columns, fill_value=0.0)
    preds = np.maximum(model.predict(X_val), MIN_LAMBDA)

    n = len(val_df)
    lam_home, lam_away = preds[:n], preds[n:]
    return [
        pb.models.create_dixon_coles_grid(float(h), float(a), rho=0.0, max_goals=max_goals)
        for h, a in zip(lam_home, lam_away)
    ]


COVARIATE_SPECS = {
    "team_effects_only": [],
    "team_effects_plus_elo_pi": ["elo_diff_signed", "pi_diff_signed"],
}


def evaluate_covariate_specs(seasons: list[str] | None = None, min_train_seasons: int = 3) -> pd.DataFrame:
    """One row per (spec, fold): every scoreline market's metrics for each
    covariate specification in `COVARIATE_SPECS`, on the same folds/shape
    `model_selection_by_segment.py` already uses (so results are directly
    comparable to Dixon-Coles/Bivariate-Poisson/ml_scoreline's own numbers
    there without re-deriving anything)."""
    folds = prepare_folds(seasons, min_train_seasons)
    rows = []
    for fold in folds:
        for spec_name, covariate_cols in COVARIATE_SPECS.items():
            model, columns, team_categories = fit(fold["train_df"], covariate_cols)
            grids = predict_grids_batch(model, columns, team_categories, fold["val_df"], covariate_cols)
            metrics = _metrics_from_grids(grids, fold["val_df"])
            rows.append({"spec": spec_name, "val_season": fold["val_season"], **metrics})
    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_fold.columns if c not in ("spec", "val_season")]
    return per_fold.groupby("spec", as_index=False)[metric_cols].mean()


if __name__ == "__main__":
    result = evaluate_covariate_specs()
    result.to_csv("reports/covariate_poisson_research.csv", index=False)
    print(result.to_string(index=False))
    print("\n--- summary ---")
    print(summarize(result).to_string(index=False))
