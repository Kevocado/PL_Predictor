"""tune_dc_bp_xi.py — grid search over Dixon-Coles/Bivariate-Poisson's
`xi` (recency-decay) parameter, across the same 5-fold chronological
walk-forward used everywhere else in this project.

Confirmed (docs/AI_CONTINUITY.md P1 #6): neither model has ever had this
tuned — both just use `xi=0.0018`, penaltyblog's own suggested default,
inherited without a search. `ml_scoreline` has a dedicated Optuna search
(`evaluate/tune_hyperparams.py`); this is the direct DC/BP analogue,
against DC/BP's one real tunable parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..data import football_data
from ..features.build import build_training_frame
from ..models import scoreline

XI_GRID = [0.0, 0.0005, 0.001, 0.0018, 0.003, 0.005, 0.008, 0.012, 0.02]

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def prepare_folds(seasons: list[str] | None = None, min_train_seasons: int = 3) -> list[dict]:
    """Same fold shape as `evaluate/walk_forward.py::prepare_folds`, minus
    the ml_scoreline-specific `X_train`/`X_val` construction — DC/BP fit
    directly on `train_df`'s goals/team columns, no feature matrix needed."""
    seasons = seasons or football_data.default_completed_seasons(n=8)
    matches_df = football_data.load_training_data(seasons=seasons)
    df, _ = build_training_frame(matches_df=matches_df)

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_df = df[df["season"].isin(seasons[:i])]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df})
    return folds


def _evaluate(model, val_df: pd.DataFrame) -> dict:
    preds = scoreline.predict_fixtures_batch(model, val_df)
    probs = np.array([[p["home_win"], p["draw"], p["away_win"]] for p in preds])
    outcomes = val_df["ftr"].map(_RESULT_CODE).to_numpy()
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
    }


def evaluate_xi_grid(model_name: str, folds: list[dict], xi_grid: list[float] = XI_GRID) -> pd.DataFrame:
    """One row per (xi, fold). `model_name` is `"dixon_coles"` or
    `"bivariate_poisson"`."""
    fit_fn = scoreline.fit_dixon_coles if model_name == "dixon_coles" else scoreline.fit_bivariate_poisson

    rows = []
    for xi in xi_grid:
        for fold in folds:
            model = fit_fn(fold["train_df"], xi=xi)
            metrics = _evaluate(model, fold["val_df"])
            rows.append({"model": model_name, "xi": xi, "val_season": fold["val_season"], **metrics})
    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    return per_fold.groupby(["model", "xi"], as_index=False)[["rps", "brier"]].mean()


if __name__ == "__main__":
    folds = prepare_folds()
    all_rows = []
    for model_name in ("dixon_coles", "bivariate_poisson"):
        result = evaluate_xi_grid(model_name, folds)
        all_rows.append(result)
    per_fold = pd.concat(all_rows, ignore_index=True)
    per_fold.to_csv("reports/dc_bp_xi_tuning.csv", index=False)

    summary = summarize(per_fold)
    print(summary.to_string(index=False))
    for model_name in ("dixon_coles", "bivariate_poisson"):
        model_summary = summary[summary["model"] == model_name]
        current = model_summary[model_summary["xi"] == 0.0018]["rps"].iloc[0]
        best_row = model_summary.loc[model_summary["rps"].idxmin()]
        print(
            f"\n{model_name}: current xi=0.0018 RPS={current:.5f}; "
            f"best xi={best_row['xi']} RPS={best_row['rps']:.5f} "
            f"(delta {current - best_row['rps']:+.5f})"
        )
