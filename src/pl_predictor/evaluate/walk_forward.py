"""walk_forward.py — multi-season walk-forward cross-validation for
ml_scoreline.

`models/manifest.py::chronological_split` always validates against exactly
one season (the last fully completed one) — a deliberate choice so retrains
stay comparable to each other over time, not a claim that one season is
enough to trust the reported RPS/Brier. This module answers that separate
question: treat each of several completed seasons as its own validation
fold, training only on strictly earlier seasons in each fold, and average
the held-out metrics across folds. That tells you how much the single
chronological-holdout number could plausibly have been an easy or hard
season, without changing what the live model actually trains on.

Deliberately NOT wired into `train_all` or the hourly auto-retrain loop —
running one fit per fold multiplies a single retrain's ~10-30s cost by the
number of folds, working against auto-retrain's "lands within about an hour
of a gameweek" design. This stays a periodic/manual confidence check (run
directly, or from a notebook).
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data
from ..features.build import build_training_frame
from ..models import ml_scoreline


def prepare_folds(seasons: list[str] | None = None, min_train_seasons: int = 3) -> list[dict]:
    """The expensive, hyperparameter-independent part of walk-forward
    validation: builds the full feature frame once and slices it into
    per-fold (train_df, val_df, ml_feature_cols) splits. Split out from
    `walk_forward_validate` so a hyperparameter search (`evaluate/
    tune_hyperparams.py`) can call `evaluate_folds` many times against the
    *same* prepared folds instead of redoing feature engineering (Elo/Pi
    replay, xG, shot-situation, pulselive fetches, ...) on every trial —
    that part doesn't depend on the model's hyperparameters at all.

    Feature engineering runs once across the full concatenated range and is
    sliced per fold rather than rebuilt per fold: `features/rolling_form.py`
    already computes every row's features from strictly that team's own
    *past* matches (`shift(1)` rolling, grouped by team, sorted by date), so
    a row's feature values never depend on which fold it lands in — the same
    no-lookahead guarantee `build_training_frame` already provides for a
    single chronological split holds per-fold here too.

    Uses the same fouls-excluded feature subset `models/manifest.py::
    train_all` actually trains `ml_scoreline` on (`ml_feature_cols`) —
    matching production exactly matters here since this is also the
    evaluation Optuna tunes against."""
    seasons = seasons or football_data.default_completed_seasons()
    matches_df = football_data.load_training_data(seasons=seasons)
    df, feature_cols = build_training_frame(matches_df=matches_df)
    ml_feature_cols = [c for c in feature_cols if "fouls" not in c]

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df})

    for fold in folds:
        fold["X_train"] = fold["train_df"][ml_feature_cols].fillna(0)
        fold["X_val"] = fold["val_df"][ml_feature_cols].fillna(0)

    return folds


def evaluate_folds(folds: list[dict], hyperparams: dict | None = None) -> pd.DataFrame:
    """One row per fold (`val_season`, `n_train`, `n_val`, `rps`, `brier`)
    for a given set of `ml_scoreline` hyperparameters, reusing folds already
    built by `prepare_folds` — no feature engineering redone here."""
    rows = []
    for fold in folds:
        train_df, val_df = fold["train_df"], fold["val_df"]
        home_model, away_model = ml_scoreline.train_goal_regressors(
            fold["X_train"], train_df["goals_home"], train_df["goals_away"], hyperparams=hyperparams
        )
        metrics = ml_scoreline.evaluate_on_holdout(home_model, away_model, fold["X_val"], val_df)
        rows.append(
            {
                "val_season": fold["val_season"],
                "n_train": len(train_df),
                "n_val": len(val_df),
                "rps": metrics["rps"],
                "brier": metrics["brier"],
            }
        )
    return pd.DataFrame(rows)


def walk_forward_validate(
    seasons: list[str] | None = None, min_train_seasons: int = 3, hyperparams: dict | None = None
) -> pd.DataFrame:
    """Convenience one-shot wrapper: `prepare_folds` + `evaluate_folds` for
    a single hyperparameter set. `hyperparams` overrides `ml_scoreline.
    DEFAULT_HYPERPARAMS` (e.g. for a one-off check) — omit for production
    defaults. Prefer calling `prepare_folds` once yourself and reusing it
    with `evaluate_folds` when evaluating many hyperparameter sets (see
    `evaluate/tune_hyperparams.py`) — this wrapper redoes feature
    engineering on every call, which is fine for a single check but wasteful
    across many."""
    folds = prepare_folds(seasons, min_train_seasons)
    return evaluate_folds(folds, hyperparams)
