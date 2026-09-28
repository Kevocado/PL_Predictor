"""blend_validation.py — does the fitted direct-vs-union mixture actually beat the
`max(...)` composite it replaced?

Context. `models/player_goals.py` originally combined the direct G+A classifier
and the Poisson union with

    max(direct_contribution, anytime_goal_prob, anytime_assist_prob)

and now uses a convex mixture instead, whose share is grid-searched by
`_fit_blend_weight` on the newest completed FPL season. That change was made on
statistical grounds (a `max` of separately-calibrated estimators is not a
calibrated estimator of anything, and `max(a, b) = (a+b)/2 + |a-b|/2` biases
upward by half the inter-model disagreement, so it shrinks least toward the base
rate exactly for the players the two models disagree most about) — but on
*evidence*. This module is the evidence.

Three design points, each forced by a specific weakness in the thing being
tested:

1. **`max(...)` is a scored arm, not the thing being replaced silently.**
   Comparing the mixture against the union alone cannot answer whether it is an
   improvement; only comparing against the incumbent can.

2. **The blend weight is fitted on a slice it is not scored on.**
   Production's `_fit_blend_weight` fits the union's Platt calibrator, grid-searches
   the weight, *and* scores by Brier on the same single season. Three fits, one
   slice. Here the calibration season is split chronologically in half: the
   calibrators are fitted on the first half, the weight is grid-searched on the
   second, and only the untouched test season is scored. Nothing about the
   reported number shaped it.

3. **The fold split is the existing one** — `train` (minus the calibration
   season) fits the classifier, the calibration season's first half fits both
   Platt calibrators, its second half fits the blend weight, and `test` is scored.
   Identical chronology to `goal_contribution_research._evaluate_fold`, so the
   `direct_enhanced` arm here is directly comparable to that module's number.

Promotion still requires the project's two-gate rule (improve the walk-forward
mean AND hold on the single most recent season) on top of beating the incumbent
arm. Neither is automatic here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import player_form
from .goal_contribution_research import (
    BASE_FEATURES,
    ENHANCED_FEATURES,
    _apply_platt,
    _fit_platt,
    _metrics,
    _poisson_union,
    build_goal_contribution_frame,
)

WEIGHT_GRID = np.linspace(0.0, 1.0, 21)

ARMS = (
    "prevalence",
    "poisson_union_calibrated",
    "direct_enhanced",
    "incumbent_max",
    "blend_mixture",
)


def _poisson_components(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The two per-component Poisson probabilities production serves.

    Mirrors `player_goals.predict_player` exactly: lambda is the rolling per-90
    rate scaled by expected minutes over 90, and `anytime_probability(lam)` is
    `1 - exp(-lam)`. Returns (P(goal), P(assist)).
    """
    minutes = frame["expected_minutes_pre_match"].to_numpy()
    scale = np.clip(minutes / 90.0, 0, None)
    lam_goals = frame["goals_per90_last10"].to_numpy() * scale
    lam_assists = frame["assists_per90_last10"].to_numpy() * scale
    return 1 - np.exp(-np.clip(lam_goals, 0, None)), 1 - np.exp(-np.clip(lam_assists, 0, None))


def _fit_blend_weight_on(
    actual: np.ndarray, direct: np.ndarray, union: np.ndarray
) -> float:
    """Grid-search the direct-model share that minimises Brier on this slice.

    Deliberately fitted and scored on the same rows — that is legitimate here
    because this slice is a *dedicated* weight-fitting slice, disjoint from both
    the calibrator-fitting slice and the test season. Production's version does
    not have that separation; see the module docstring.
    """
    from sklearn.metrics import brier_score_loss

    best_weight, best_score = 1.0, float("inf")
    for weight in WEIGHT_GRID:
        score = brier_score_loss(actual, weight * direct + (1.0 - weight) * union)
        if score < best_score:
            best_score, best_weight = score, float(weight)
    return best_weight


def evaluate_blend_arms(
    seasons: list[str] | None = None, min_train_seasons: int = 3
) -> dict:
    """Walk forward and score every arm on an untouched test season.

    Returns a report-shaped dict; callers must inspect the metrics. Nothing here
    is fitted for production or written to `models/`.
    """
    frame, _ = build_goal_contribution_frame(seasons)
    available = sorted(frame["season"].unique())

    feature_names = [f for f in BASE_FEATURES + ENHANCED_FEATURES if f in frame] + ["position"]
    rows: list[dict] = []
    weights: list[dict] = []

    for index in range(min_train_seasons, len(available)):
        test_season = available[index]
        train_seasons = available[:index]

        train = frame[frame["season"].isin(train_seasons)]
        test = frame[frame["season"] == test_season]
        calibration_season = train_seasons[-1]
        fit = train[train["season"] != calibration_season]
        calibration = train[train["season"] == calibration_season].sort_values(
            ["element", "kickoff_time"]
        )
        if fit.empty or calibration.empty or test.empty:
            continue

        # Split the calibration season chronologically. `index // 2` is a row
        # position, not a date boundary, so it follows the existing sort order.
        midpoint = len(calibration) // 2
        calibrator_fit = calibration.iloc[:midpoint]
        weight_fit = calibration.iloc[midpoint:]
        if calibrator_fit.empty or weight_fit.empty:
            continue
        y_cal = calibrator_fit["goal_contribution"].to_numpy()
        y_weight = weight_fit["goal_contribution"].to_numpy()
        y_fit = fit["goal_contribution"].to_numpy()
        y_test = test["goal_contribution"].to_numpy()
        if len(np.unique(y_cal)) < 2 or len(np.unique(y_weight)) < 2:
            continue

        def design(rows: pd.DataFrame, columns: list[str] | None = None):
            matrix = pd.get_dummies(rows[feature_names], columns=["position"], dtype=float)
            matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if columns is not None:
                matrix = matrix.reindex(columns=columns, fill_value=0.0)
            return matrix, matrix.columns.tolist()

        X_fit, columns = design(fit)
        X_cal, _ = design(calibrator_fit, columns)
        X_weight, _ = design(weight_fit, columns)
        X_test, _ = design(test, columns)

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X_fit, y_fit)

        # Direct arm: Platt calibrator fitted on calibrator_fit only.
        direct_calibrator = _fit_platt(model.predict_proba(X_cal)[:, 1], y_cal)
        direct_weight = _apply_platt(direct_calibrator, model.predict_proba(X_weight)[:, 1])
        direct_test = _apply_platt(direct_calibrator, model.predict_proba(X_test)[:, 1])

        # Union arm: its own Platt calibrator, also fitted on calibrator_fit only.
        union_calibrator = _fit_platt(_poisson_union(calibrator_fit), y_cal)
        union_weight = _apply_platt(union_calibrator, _poisson_union(weight_fit))
        union_test = _apply_platt(union_calibrator, _poisson_union(test))

        # Weight fitted on weight_fit, scored on test. Never fitted on test.
        weight = _fit_blend_weight_on(y_weight, direct_weight, union_weight)
        weights.append({"test_season": test_season, "blend_weight": weight})

        # Incumbent: the max(...) composite production used to serve, verbatim.
        goal_prob, assist_prob = _poisson_components(test)
        incumbent = np.maximum(np.maximum(direct_test, goal_prob), assist_prob)

        blend = weight * direct_test + (1.0 - weight) * union_test

        for arm, probability in (
            ("prevalence", np.full(len(y_test), y_fit.mean())),
            ("poisson_union_calibrated", union_test),
            ("direct_enhanced", direct_test),
            ("incumbent_max", incumbent),
            ("blend_mixture", blend),
        ):
            rows.append(
                {"fold": test_season, "model": arm, "n_test": len(test), **_metrics(y_test, probability)}
            )

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model", as_index=False)[["brier", "log_loss", "average_precision", "ece"]].mean()
        if not metrics.empty
        else pd.DataFrame()
    )
    return {"metrics": metrics, "summary": summary, "weights": pd.DataFrame(weights)}


def _fold_table(metrics: pd.DataFrame) -> str:
    pivots = {}
    for arm in ARMS:
        sub = metrics[metrics["model"] == arm]
        if not sub.empty:
            pivots[arm] = sub.set_index("fold")["brier"]
    return pd.DataFrame(pivots).round(6).to_string()


if __name__ == "__main__":
    result = evaluate_blend_arms()
    print("=== per-fold Brier (lower is better) ===")
    print(_fold_table(result["metrics"]))
    print("\n=== walk-forward mean ===")
    print(result["summary"].round(6).to_string(index=False))
    print("\n=== fitted blend weight per fold (fitted on a slice disjoint from test) ===")
    print(result["weights"].to_string(index=False))
