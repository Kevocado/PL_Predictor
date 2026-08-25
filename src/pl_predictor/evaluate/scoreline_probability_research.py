"""Leakage-safe calibration and ensemble experiments for scoreline probabilities.

This module intentionally does not alter the live model. Each outer fold keeps
one future season untouched, reserves the latest earlier season for calibration,
and fits every base model only on the seasons before that. It then compares the
raw ML scoreline probabilities, multinomial Platt scaling, and a constrained
blend of ML/Dixon-Coles/Bivariate-Poisson on the untouched future season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..data import football_data
from ..features.build import build_training_frame
from ..models import ml_scoreline, scoreline

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def _ece(outcomes: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Mean class-wise expected calibration error for 1X2 probabilities."""
    errors = []
    for class_index in range(probabilities.shape[1]):
        predicted = probabilities[:, class_index]
        actual = (outcomes == class_index).astype(float)
        bucket = np.clip((predicted * bins).astype(int), 0, bins - 1)
        errors.append(
            sum(
                abs(actual[bucket == index].mean() - predicted[bucket == index].mean())
                * (bucket == index).mean()
                for index in range(bins)
                if (bucket == index).any()
            )
        )
    return float(np.mean(errors))


def _metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return {
        "rps": float(pb.metrics.rps_average(probabilities, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probabilities, outcomes)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
        "ece": _ece(outcomes, probabilities),
    }


def _matrix_from_ml(home_model, away_model, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, frame[feature_cols].fillna(0))
    return np.array([[grid.home_win, grid.draw, grid.away_win] for grid in grids])


def _matrix_from_goal_model(model, frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            [prediction["home_win"], prediction["draw"], prediction["away_win"]]
            for prediction in (scoreline.predict_fixture(model, row.team_home, row.team_away) for row in frame.itertuples())
        ]
    )


def _log_probability_features(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, 1e-6, 1.0))


def _fit_platt(calibration_probabilities: np.ndarray, outcomes: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1_000, C=0.25)
    model.fit(_log_probability_features(calibration_probabilities), outcomes)
    return model


def _blend_weights(probability_sets: list[np.ndarray], outcomes: np.ndarray) -> np.ndarray:
    stacked = np.stack(probability_sets, axis=1)

    def objective(weights):
        blended = np.tensordot(stacked, weights, axes=(1, 0))
        return np.mean(np.sum((blended - np.eye(3)[outcomes]) ** 2, axis=1))

    result = minimize(
        objective,
        x0=np.full(len(probability_sets), 1 / len(probability_sets)),
        bounds=[(0.0, 1.0)] * len(probability_sets),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        method="SLSQP",
    )
    return result.x if result.success else np.full(len(probability_sets), 1 / len(probability_sets))


def run_scoreline_probability_research(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    """Return all chronological fold metrics for calibration and blending.

    The earliest train season in each outer fold is used for model fitting; the
    latest is reserved for calibration/weight fitting. This costs some fitting
    data, deliberately, to keep the future evaluation season completely clean.
    """
    seasons = seasons or football_data.default_completed_seasons()
    matches = football_data.load_training_data(seasons=seasons)
    frame, feature_cols = build_training_frame(matches_df=matches)
    ml_feature_cols = [column for column in feature_cols if "fouls" not in column]
    rows = []

    for season_index in range(min_train_seasons, len(seasons)):
        fit_seasons = seasons[: season_index - 1]
        calibration_season = seasons[season_index - 1]
        validation_season = seasons[season_index]
        fit = frame[frame["season"].isin(fit_seasons)]
        calibration = frame[frame["season"] == calibration_season]
        validation = frame[frame["season"] == validation_season]
        if fit.empty or calibration.empty or validation.empty:
            continue

        home_model, away_model = ml_scoreline.train_goal_regressors(
            fit[ml_feature_cols].fillna(0), fit["goals_home"], fit["goals_away"]
        )
        dixon_coles = scoreline.fit_dixon_coles(fit)
        bivariate_poisson = scoreline.fit_bivariate_poisson(fit)

        calibration_outcomes = calibration["ftr"].map(RESULT_CODE).to_numpy()
        validation_outcomes = validation["ftr"].map(RESULT_CODE).to_numpy()
        calibration_sets = [
            _matrix_from_ml(home_model, away_model, calibration, ml_feature_cols),
            _matrix_from_goal_model(dixon_coles, calibration),
            _matrix_from_goal_model(bivariate_poisson, calibration),
        ]
        validation_sets = [
            _matrix_from_ml(home_model, away_model, validation, ml_feature_cols),
            _matrix_from_goal_model(dixon_coles, validation),
            _matrix_from_goal_model(bivariate_poisson, validation),
        ]
        platt = _fit_platt(calibration_sets[0], calibration_outcomes)
        weights = _blend_weights(calibration_sets, calibration_outcomes)
        candidates = {
            "ml_uncalibrated": validation_sets[0],
            "ml_platt": platt.predict_proba(_log_probability_features(validation_sets[0])),
            "scoreline_blend": np.tensordot(np.stack(validation_sets, axis=1), weights, axes=(1, 0)),
        }
        for name, probabilities in candidates.items():
            rows.append(
                {
                    "fold": validation_season,
                    "calibration_season": calibration_season,
                    "model": name,
                    "n_fit": len(fit),
                    "n_calibration": len(calibration),
                    "n_validation": len(validation),
                    "weights": weights.tolist() if name == "scoreline_blend" else None,
                    **_metrics(validation_outcomes, probabilities),
                }
            )

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model", as_index=False)[["rps", "brier", "log_loss", "ece"]].mean().sort_values("brier")
        if not metrics.empty
        else pd.DataFrame(columns=["model", "rps", "brier", "log_loss", "ece"])
    )
    return {"metrics": metrics, "summary": summary}


if __name__ == "__main__":
    report = run_scoreline_probability_research()
    print(report["summary"].to_string(index=False))
